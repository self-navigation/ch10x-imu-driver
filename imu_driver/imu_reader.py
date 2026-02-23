#!/usr/bin/env python3
"""
CH100 IMU ROS2 Driver Node
(With updates from C++ upstream https://github.com/hipnuc/products/tree/dddc5d46add235d848d860a326eab8ee5f66c919/examples/ROS2)

Reads binary 0x91 IMUSOL frames from the CH100 over serial and publishes:
  - imu/data          (sensor_msgs/Imu)        - quaternion, angular vel, linear accel
  - imu/mag           (sensor_msgs/MagneticField) - magnetometer XYZ
  - imu/pressure      (sensor_msgs/FluidPressure)  - barometer
  - imu/euler         (geometry_msgs/Vector3Stamped) - roll/pitch/yaw in radians (debug)

The CH100 uses:
  Body frame:  Right-Forward-Up (RFU)  - compatible with ROS REP-103
  World frame: East-North-Up   (ENU)  - compatible with ROS REP-103

Frame wire format (82 bytes total):
  [0x5A][0xA5][LEN_L][LEN_H][CRC_L][CRC_H][ ... 76-byte payload ... ]
  CRC-16/CCITT (poly=0x1021, init=0x0000) over bytes 0..3 + payload.

Payload (0x91 packet, 76 bytes):
  Offset  Type       Size  Unit       Field
  0       uint8      1     -          Packet label (0x91)
  1-2     uint16     2     -          Main status flags
  3       int8       1     °C         Temperature
  4-7     float32    4     Pa         Air pressure
  8-11    uint32     4     ms         Timestamp since power-on
  12-23   float32×3  12    G          Acceleration XYZ  (1G = 9.80665 m/s²)
  24-35   float32×3  12    °/s        Angular velocity XYZ
  36-47   float32×3  12    µT         Magnetic field XYZ
  48-59   float32×3  12    °          Euler angles: Roll, Pitch, Yaw
  60-75   float32×4  16    -          Quaternion: W, X, Y, Z
"""

import struct
import serial
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Imu, MagneticField, FluidPressure
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Header

import math
import queue
import threading

from rclpy.time import Time

# -----------------------------------------------------------------------------
# Protocol constants
# -----------------------------------------------------------------------------
FRAME_HEADER_1 = 0x5A
FRAME_HEADER_2 = 0xA5
PAYLOAD_LABEL = 0x91
HEADER_SIZE = 6  # 0x5A + 0xA5 + 2-byte LEN + 2-byte CRC
PAYLOAD_SIZE = 76  # fixed for 0x91 packet
TOTAL_FRAME_SIZE = HEADER_SIZE + PAYLOAD_SIZE  # 82 bytes

G_TO_MS2 = 9.80665  # 1 G in m/s²
DEG_TO_RAD = math.pi / 180.0
UT_TO_T = 1e-6  # µT → Tesla

# Covariance matrices - adjust these to match your calibration data.
# Using -1.0 as the first element means "unknown" per REP-147.
ORIENTATION_COVAR = [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4]

ANGULAR_VEL_COVAR = [1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6]

LINEAR_ACCEL_COVAR = [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4]

MAG_COVAR = [1e-9, 0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, 0.0, 1e-9]


# -----------------------------------------------------------------------------
# CRC-16/CCITT  (polynomial 0x1021, initial value 0x0000)
# Matches the C implementation in the CH100 manual exactly.
# -----------------------------------------------------------------------------
def crc16_update(crc: int, data: bytes) -> int:
    """Update a running CRC-16/CCITT value with new bytes."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            temp = crc << 1
            if crc & 0x8000:
                temp ^= 0x1021
            crc = temp & 0xFFFF  # keep to 16 bits
    return crc


# -----------------------------------------------------------------------------
# HiPNUC frame decoder - byte-by-byte state machine
# https://github.com/hipnuc/products/blob/dddc5d46add235d848d860a326eab8ee5f66c919/examples/ROS2/hipnuc_ws/src/hipnuc_lib_package/src/hipnuc_dec.c
#
# Direct Python translation of hipnuc_input() + decode_hipnuc() from
# hipnuc_dec.c.  Feed one byte at a time via input(); it returns the
# raw payload bytes when a complete, CRC-verified frame has arrived,
# or None otherwise.
# -----------------------------------------------------------------------------
class HipnucDecoder:
    _SYNC1 = 0x5A
    _SYNC2 = 0xA5
    _HEADER_SIZE = 6  # sync(2) + len(2) + crc(2)
    _MAX_SIZE = 512  # matches HIPNUC_MAX_RAW_SIZE in hipnuc_dec.h

    def __init__(self):
        # Internal accumulation buffer - pre-allocated like the C struct
        self._buf = bytearray(self._MAX_SIZE)
        # Number of bytes currently buffered (0 means waiting for sync)
        self._nbyte = 0
        # Declared payload length extracted from the length field
        self._len = 0

    def input(self, byte: int) -> bytes | None:
        """
        Process one byte from the serial stream.

        Returns:
            bytes: the validated payload (length == self._len) if this byte
                   completed a good frame.
            None:  still accumulating, or the frame failed validation.
        """
        if self._nbyte == 0:
            # -- Sync phase: maintain a 2-byte sliding window --------------
            # Equivalent to sync_hipnuc() in hipnuc_dec.c:
            #   buf[0] = buf[1]; buf[1] = data;
            #   return buf[0] == CHSYNC1 && buf[1] == CHSYNC2;
            self._buf[0] = self._buf[1]
            self._buf[1] = byte
            if self._buf[0] == self._SYNC1 and self._buf[1] == self._SYNC2:
                # Sync found - start accumulating from byte 2
                self._nbyte = 2
            return None

        # -- Accumulation phase --------------------------------------------
        self._buf[self._nbyte] = byte
        self._nbyte += 1

        # Once we have the full 6-byte header, extract the payload length
        if self._nbyte == self._HEADER_SIZE:
            self._len = self._buf[2] | (self._buf[3] << 8)
            if self._len > self._MAX_SIZE - self._HEADER_SIZE:
                # Declared length is insane - almost certainly a missed sync
                self._nbyte = 0
                return None

        # Keep waiting until the complete frame (header + payload) is buffered
        if self._nbyte < self._HEADER_SIZE:
            return None
        if self._nbyte < self._HEADER_SIZE + self._len:
            return None

        # -- Frame complete - verify CRC then hand off payload -------------
        # Reset nbyte first so re-sync starts immediately on any failure
        self._nbyte = 0

        return self._verify_and_extract()

    def _verify_and_extract(self) -> bytes | None:
        """
        Verify CRC and return the payload slice, or None on failure.

        CRC covers:
          buf[0..3]  - sync bytes + length field  (excludes the CRC field itself)
          buf[6..]   - payload
        This matches decode_hipnuc() in hipnuc_dec.c exactly.
        """
        payload_len = self._len

        # Compute CRC over header bytes 0..3 and the full payload
        crc = crc16_update(0, bytes(self._buf[:4]))
        crc = crc16_update(
            crc, bytes(self._buf[self._HEADER_SIZE : self._HEADER_SIZE + payload_len])
        )

        # CRC is stored at bytes 4..5, little-endian
        frame_crc = self._buf[4] | (self._buf[5] << 8)

        if crc != frame_crc:
            return None  # caller increments bad-frame counter

        return bytes(self._buf[self._HEADER_SIZE : self._HEADER_SIZE + payload_len])


# -----------------------------------------------------------------------------
# Packet parser
# -----------------------------------------------------------------------------
def parse_payload(payload: bytes) -> dict:
    """
    Unpack a 76-byte 0x91 IMUSOL payload into a plain dict.
    All unit conversions are done here so the ROS node stays clean.
    """
    # Struct layout (little-endian), matches hi91_t in hipnuc_dec.h exactly:
    # B  = uint8   (label/tag)
    # H  = uint16  (main_status)
    # b  = int8    (temperature °C)
    # f  = float32 (pressure Pa)
    # I  = uint32  (timestamp ms)
    # 3f = float32 × 3 (accel G)
    # 3f = float32 × 3 (gyro °/s)
    # 3f = float32 × 3 (mag µT)
    # 3f = float32 × 3 (euler °: roll, pitch, yaw)
    # 4f = float32 × 4 (quaternion: w, x, y, z)
    fmt = "<BHbfI3f3f3f3f4f"
    expected = struct.calcsize(fmt)  # should be 76
    assert (
        expected == PAYLOAD_SIZE
    ), f"Struct size mismatch: {expected} != {PAYLOAD_SIZE}"

    fields = struct.unpack(fmt, payload)
    (
        label,
        main_status,
        temp_c,
        pressure_pa,
        timestamp_ms,
        ax,
        ay,
        az,
        gx,
        gy,
        gz,
        mx,
        my,
        mz,
        roll_deg,
        pitch_deg,
        yaw_deg,
        qw,
        qx,
        qy,
        qz,
    ) = fields

    return {
        "label": label,
        "main_status": main_status,
        "temp_c": temp_c,
        # Pressure - keep as Pa (SI)
        "pressure_pa": pressure_pa,
        # Timestamp in seconds
        "timestamp_s": timestamp_ms * 1e-3,
        # Acceleration: G → m/s²
        "accel_ms2": (ax * G_TO_MS2, ay * G_TO_MS2, az * G_TO_MS2),
        # Angular velocity: °/s → rad/s
        "gyro_rads": (gx * DEG_TO_RAD, gy * DEG_TO_RAD, gz * DEG_TO_RAD),
        # Magnetometer: µT → Tesla
        "mag_t": (mx * UT_TO_T, my * UT_TO_T, mz * UT_TO_T),
        # Euler angles: ° → rad (for convenience; quaternion is preferred)
        "euler_rad": (
            roll_deg * DEG_TO_RAD,
            pitch_deg * DEG_TO_RAD,
            yaw_deg * DEG_TO_RAD,
        ),
        # Quaternion (already dimensionless)
        "quat_wxyz": (qw, qx, qy, qz),
    }


# -----------------------------------------------------------------------------
# Hardware clock synchroniser
# -----------------------------------------------------------------------------
class HwClockSync:
    """
    Maps the IMU's internal millisecond counter to ROS wall-clock time.

    The IMU reports a uint32 'system_time' field (ms since power-on) that
    increments at a rate governed by its own crystal oscillator.  This clock
    is precise (low jitter) but its frequency differs from the host CPU clock
    by up to ±50 ppm, so a naive one-shot offset computed at startup drifts
    by up to ~4 ms/min.

    Strategy - offset EMA with a residual gate:

      offset_ns  =  ros_now_ns  −  hw_ns          (new observation)
      filtered   +=  α × (offset_ns − filtered)   (EMA update)
      stamp      =  hw_ns + filtered

    The EMA smooths per-frame OS scheduling jitter while tracking the slow
    crystal frequency error.  The residual gate skips the EMA update when
    the residual exceeds one nominal frame period: this protects the estimate
    against moments when the serial byte sat in the OS buffer longer than
    usual (high-latency reads), which would otherwise bias the offset high.

    Wraparound: uint32 ms wraps at 2^32 ms ≈ 49.7 days.  Detected by
    checking whether hw_ms jumped backwards by more than 1 second relative
    to the previous frame.
    """

    # EMA coefficient - time constant ≈ 1/α frames.
    # At 100 Hz → ~5 s settling; at 400 Hz → ~1.25 s settling.
    _ALPHA = 0.002

    # Skip EMA update if the observed offset deviates from the filtered value
    # by more than this - indicates high serial-read latency, not true drift.
    _MAX_RESIDUAL_NS = 5_000_000  # 5 ms

    # uint32 ms counter full range in nanoseconds (for wraparound compensation)
    _WRAP_PERIOD_NS = (1 << 32) * 1_000_000

    def __init__(self):
        self._filtered_offset_ns: float | None = None
        self._prev_hw_ms: int = 0
        self._wrap_count: int = 0

    def stamp_ns(self, hw_ms: int, ros_now_ns: int) -> int:
        """
        Returns a corrected ROS timestamp in nanoseconds.

        hw_ms      - raw uint32 millisecond counter from the IMU payload
        ros_now_ns - result of node.get_clock().now().nanoseconds, taken
                     immediately after the last byte of the frame was decoded
        """
        # -- Wraparound detection -----------------------------------------
        # Consecutive frames are at most a few ms apart, so a backwards jump
        # of more than 1 second can only mean the uint32 counter wrapped.
        if self._prev_hw_ms > 0 and hw_ms < self._prev_hw_ms - 1000:
            self._wrap_count += 1
        self._prev_hw_ms = hw_ms

        hw_ns = hw_ms * 1_000_000 + self._wrap_count * self._WRAP_PERIOD_NS

        # -- Bootstrap on first frame -------------------------------------
        if self._filtered_offset_ns is None:
            self._filtered_offset_ns = float(ros_now_ns - hw_ns)
            return ros_now_ns  # first frame: use wall clock directly

        # -- EMA update with residual gate --------------------------------
        observed_offset = ros_now_ns - hw_ns
        residual = observed_offset - self._filtered_offset_ns
        if abs(residual) < self._MAX_RESIDUAL_NS:
            self._filtered_offset_ns += self._ALPHA * residual

        return hw_ns + int(self._filtered_offset_ns)


# -----------------------------------------------------------------------------
# Hardware timestamp extraction helper
# -----------------------------------------------------------------------------
# timestamp_ms sits at payload byte offset 8 in the 0x91 packet:
#   B(1) + H(2) + b(1) + f(4) = 8 bytes before the uint32 field.
# Extracted here without a full struct.unpack so the reader thread
# can timestamp frames before the payload is fully parsed.
_HW_TS_OFFSET = struct.calcsize("<BHbf")  # == 8
_HW_TS_FMT = struct.Struct("<I")  # uint32, little-endian


def _extract_hw_ms(payload: bytes) -> int:
    """Return the raw uint32 millisecond timestamp from a 0x91 payload."""
    return _HW_TS_FMT.unpack_from(payload, _HW_TS_OFFSET)[0]


# -----------------------------------------------------------------------------
# ROS2 Node
# -----------------------------------------------------------------------------
class CH100ImuNode(Node):

    def __init__(self):
        super().__init__("ch100_imu_node")

        # -- Declare parameters (can be overridden via launch args or YAML) --
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("publish_euler", False)  # debug topic

        port = self.get_parameter("port").value
        baud_rate = self.get_parameter("baud_rate").value
        self.frame_id_ = self.get_parameter("frame_id").value
        self.pub_euler_ = self.get_parameter("publish_euler").value

        # -- QoS - sensor data: best-effort, keep last 10 --
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # -- Publishers --
        self.pub_imu_ = self.create_publisher(Imu, "imu/data", qos)
        self.pub_mag_ = self.create_publisher(MagneticField, "imu/mag", qos)
        self.pub_pressure_ = self.create_publisher(FluidPressure, "imu/pressure", qos)
        if self.pub_euler_:
            self.pub_euler_msg_ = self.create_publisher(
                Vector3Stamped, "imu/euler", qos
            )

        # -- Open serial port --
        try:
            self.serial_ = serial.Serial(
                port=port,
                baudrate=baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0,  # 1 s read timeout
            )
            self.get_logger().info(f"Opened serial port {port} at {baud_rate} baud.")
        except serial.SerialException as e:
            self.get_logger().fatal(f"Failed to open serial port: {e}")
            raise

        # -- Frame decoder (state machine - owned by the reader thread) --
        self.decoder_ = HipnucDecoder()

        # -- Hardware clock synchroniser ----------------------------------
        # Maps the IMU's internal ms counter to ROS time via a calibrated
        # offset with EMA drift tracking.  See HwClockSync for details.
        self._hw_clock_sync = HwClockSync()

        # -- Statistics --
        self.frames_ok_ = 0
        self.frames_bad_ = 0

        # -- Thread-safe queue: reader thread → ROS executor --------------
        # Each entry is a (payload: bytes, stamp: Time) tuple, stamped at
        # the moment the complete frame was decoded in the reader thread,
        # as close to receipt as possible.
        self._frame_queue = queue.SimpleQueue()

        # -- Stop event - set by destroy_node() to unblock the reader thread --
        self._stop_event = threading.Event()

        # -- Serial reader thread ------------------------------------------
        # Blocks on serial_.read(1) which releases the GIL, so the ROS
        # executor continues running freely in the main thread.
        self._reader_thread = threading.Thread(
            target=self._serial_reader,
            name="ch100_serial_reader",
            daemon=True,
        )
        self._reader_thread.start()

        # -- Publisher timer - drains the frame queue in the executor thread --
        # Period is set to match the maximum ODR (400 Hz → 2.5 ms).
        self.timer_ = self.create_timer(0.0025, self._publish_pending)

        self.get_logger().info("CH100 IMU node started.")

    # --------------------------------------------------------------------------
    def _serial_reader(self):
        """
        Runs in a dedicated background thread for the lifetime of the node.

        Reads the serial port one byte at a time and feeds each byte into the
        HipnucDecoder state machine.  serial_.read(1) blocks until a byte
        arrives, releasing the GIL so the ROS executor runs freely in the
        main thread.

        When a complete, CRC-verified frame is decoded, the payload and a ROS
        timestamp taken at that moment are pushed onto _frame_queue for the
        publisher timer to consume.
        """
        while not self._stop_event.is_set():
            try:
                data = self.serial_.read(1)
                if not data:
                    continue  # read timeout (serial timeout=1.0 s) - loop and recheck

                payload = self.decoder_.input(data[0])
                if payload is None:
                    continue  # state machine still accumulating

                # -- Complete, CRC-verified frame --------------------------
                # Wall-clock sample taken immediately after the last byte was
                # decoded - as close to receipt as possible.  Passed to
                # HwClockSync together with the IMU's own hardware timestamp
                # to produce a corrected, jitter-filtered ROS stamp.
                ros_now_ns = self.get_clock().now().nanoseconds

                if len(payload) != PAYLOAD_SIZE:
                    self.get_logger().warn(
                        f"Unexpected payload length {len(payload)}, "
                        f"expected {PAYLOAD_SIZE} - skipping."
                    )
                    self.frames_bad_ += 1
                    continue

                if payload[0] != PAYLOAD_LABEL:
                    self.get_logger().warn(
                        f"Unexpected packet label 0x{payload[0]:02X} - skipping."
                    )
                    self.frames_bad_ += 1
                    continue

                hw_ms = _extract_hw_ms(payload)
                corrected_ns = self._hw_clock_sync.stamp_ns(hw_ms, ros_now_ns)
                stamp = Time(nanoseconds=corrected_ns).to_msg()
                self._frame_queue.put((payload, stamp))

            except serial.SerialException as e:
                if not self._stop_event.is_set():
                    self.get_logger().error(f"Serial error in reader thread: {e}")
                break

    # --------------------------------------------------------------------------
    def _publish_pending(self):
        """
        Called by the ROS timer in the executor thread every 2.5 ms.

        Drains all frames that the reader thread has queued since the last
        tick and publishes them.  Under normal operation at ≤400 Hz this
        processes exactly one frame per call; after a scheduling hiccup it
        may process several.
        """
        while not self._frame_queue.empty():
            payload, stamp = self._frame_queue.get()

            data = parse_payload(payload)
            self._publish_imu(data, stamp)
            self._publish_mag(data, stamp)
            self._publish_pressure(data, stamp)
            if self.pub_euler_:
                self._publish_euler(data, stamp)

            self.frames_ok_ += 1
            if self.frames_ok_ % 500 == 0:
                self.get_logger().info(
                    f"Frames OK: {self.frames_ok_}  Bad: {self.frames_bad_}"
                )

    # --------------------------------------------------------------------------
    def _make_header(self, stamp) -> Header:
        h = Header()
        h.stamp = stamp
        h.frame_id = self.frame_id_
        return h

    # --------------------------------------------------------------------------
    def _publish_imu(self, data: dict, stamp):
        msg = Imu()
        msg.header = self._make_header(stamp)

        qw, qx, qy, qz = data["quat_wxyz"]
        msg.orientation.w = qw
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation_covariance = ORIENTATION_COVAR

        gx, gy, gz = data["gyro_rads"]
        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = gz
        msg.angular_velocity_covariance = ANGULAR_VEL_COVAR

        ax, ay, az = data["accel_ms2"]
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az
        msg.linear_acceleration_covariance = LINEAR_ACCEL_COVAR

        self.pub_imu_.publish(msg)

    # --------------------------------------------------------------------------
    def _publish_mag(self, data: dict, stamp):
        msg = MagneticField()
        msg.header = self._make_header(stamp)

        mx, my, mz = data["mag_t"]
        msg.magnetic_field.x = mx
        msg.magnetic_field.y = my
        msg.magnetic_field.z = mz
        msg.magnetic_field_covariance = MAG_COVAR

        self.pub_mag_.publish(msg)

    # --------------------------------------------------------------------------
    def _publish_pressure(self, data: dict, stamp):
        msg = FluidPressure()
        msg.header = self._make_header(stamp)
        msg.fluid_pressure = data["pressure_pa"]  # already in Pa
        msg.variance = 0.0  # unknown; set if calibrated
        self.pub_pressure_.publish(msg)

    # --------------------------------------------------------------------------
    def _publish_euler(self, data: dict, stamp):
        """Convenience debug topic: roll / pitch / yaw in radians."""
        msg = Vector3Stamped()
        msg.header = self._make_header(stamp)
        roll, pitch, yaw = data["euler_rad"]
        msg.vector.x = roll
        msg.vector.y = pitch
        msg.vector.z = yaw
        self.pub_euler_msg_.publish(msg)

    # --------------------------------------------------------------------------
    def destroy_node(self):
        """Clean up the reader thread and serial port on shutdown."""
        # Signal the reader thread to stop, then wait for it to exit.
        # The thread unblocks within serial timeout (1.0 s) at most.
        self._stop_event.set()
        if hasattr(self, "_reader_thread"):
            self._reader_thread.join(timeout=2.0)
        if hasattr(self, "serial_") and self.serial_.is_open:
            self.serial_.close()
            self.get_logger().info("Serial port closed.")
        super().destroy_node()


# -----------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = CH100ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
