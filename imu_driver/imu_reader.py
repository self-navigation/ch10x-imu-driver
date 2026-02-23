#!/usr/bin/env python3
"""
CH100 IMU ROS2 Driver Node
Reads binary 0x91 IMUSOL frames from the CH100 over serial and publishes:
  - /imu/data          (sensor_msgs/Imu)        — quaternion, angular vel, linear accel
  - /imu/mag           (sensor_msgs/MagneticField) — magnetometer XYZ
  - /imu/pressure      (sensor_msgs/FluidPressure)  — barometer
  - /imu/euler         (geometry_msgs/Vector3Stamped) — roll/pitch/yaw in radians (debug)

The CH100 uses:
  Body frame:  Right-Forward-Up (RFU)  — compatible with ROS REP-103
  World frame: East-North-Up   (ENU)  — compatible with ROS REP-103

Frame wire format (82 bytes total):
  [0x5A][0xA5][LEN_L][LEN_H][CRC_L][CRC_H][ ... 76-byte payload ... ]
  CRC-16/CCITT (poly=0x1021, init=0x0000) over bytes 0..3 + payload.

Payload (0x91 packet, 76 bytes):
  Offset  Type       Size  Unit       Field
  0       uint8      1     —          Packet label (0x91)
  1       uint8      1     —          Module ID
  2-3     —          2     —          Reserved
  4-7     float32    4     Pa         Air pressure
  8-11    uint32     4     ms         Timestamp since power-on
  12-23   float32×3  12    G          Acceleration XYZ  (1G = 9.80665 m/s²)
  24-35   float32×3  12    °/s        Angular velocity XYZ
  36-47   float32×3  12    µT         Magnetic field XYZ
  48-59   float32×3  12    °          Euler angles: Roll, Pitch, Yaw
  60-75   float32×4  16    —          Quaternion: W, X, Y, Z
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

# Covariance matrices — adjust these to match your calibration data.
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


def compute_frame_crc(header_bytes: bytes, payload_bytes: bytes) -> int:
    """
    CRC covers: frame_header + frame_type + length_field + payload.
    That is bytes [0..3] of the frame header block plus the full payload.
    The two CRC bytes themselves are excluded.
    """
    crc = crc16_update(0, header_bytes)  # 4 bytes: 0x5A 0xA5 LEN_L LEN_H
    crc = crc16_update(crc, payload_bytes)
    return crc


# -----------------------------------------------------------------------------
# Packet parser
# -----------------------------------------------------------------------------
def parse_payload(payload: bytes) -> dict:
    """
    Unpack a 76-byte 0x91 IMUSOL payload into a plain dict.
    All unit conversions are done here so the ROS node stays clean.
    """
    # Struct layout (little-endian):
    # B  = uint8   (label)
    # B  = uint8   (id)
    # 2x = 2 padding bytes
    # f  = float32 (pressure Pa)
    # I  = uint32  (timestamp ms)
    # 3f = float32 × 3 (accel G)
    # 3f = float32 × 3 (gyro °/s)
    # 3f = float32 × 3 (mag µT)
    # 3f = float32 × 3 (euler °: roll, pitch, yaw)
    # 4f = float32 × 4 (quaternion: w, x, y, z)
    fmt = "<BB2xfI3f3f3f3f4f"
    expected = struct.calcsize(fmt)  # should be 76
    assert (
        expected == PAYLOAD_SIZE
    ), f"Struct size mismatch: {expected} != {PAYLOAD_SIZE}"

    fields = struct.unpack(fmt, payload)
    (
        label,
        mod_id,
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
        "id": mod_id,
        # Pressure — keep as Pa (SI)
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
# ROS2 Node
# -----------------------------------------------------------------------------
class CH100ImuNode(Node):

    def __init__(self):
        super().__init__("ch100_imu_node")

        # -- Declare parameters (can be overridden via launch args or YAML) --
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("publish_euler", True)  # debug topic

        port = self.get_parameter("port").value
        baud_rate = self.get_parameter("baud_rate").value
        self.frame_id_ = self.get_parameter("frame_id").value
        self.pub_euler_ = self.get_parameter("publish_euler").value

        # -- QoS — sensor data: best-effort, keep last 10 --
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # -- Publishers --
        self.pub_imu_ = self.create_publisher(Imu, "/imu/data", qos)
        self.pub_mag_ = self.create_publisher(MagneticField, "/imu/mag", qos)
        self.pub_pressure_ = self.create_publisher(FluidPressure, "/imu/pressure", qos)
        if self.pub_euler_:
            self.pub_euler_msg_ = self.create_publisher(
                Vector3Stamped, "/imu/euler", qos
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

        # -- Statistics --
        self.frames_ok_ = 0
        self.frames_bad_ = 0

        # -- Timer-driven read loop (runs at ~1 kHz, i.e., faster than 400 Hz ODR) --
        self.timer_ = self.create_timer(0.001, self.read_and_publish)

        self.get_logger().info("CH100 IMU node started.")

    # --------------------------------------------------------------------------
    def read_and_publish(self):
        """Called by the ROS timer. Reads one complete frame and publishes it."""
        try:
            if not self._sync_to_frame_header():
                return  # timed-out waiting for header — try next tick

            # Read remaining 4 header bytes: LEN_L, LEN_H, CRC_L, CRC_H
            rest_header = self.serial_.read(4)
            if len(rest_header) < 4:
                self.get_logger().warn("Incomplete header read — skipping frame.")
                self.frames_bad_ += 1
                return

            payload_len = rest_header[0] | (rest_header[1] << 8)
            frame_crc = rest_header[2] | (rest_header[3] << 8)

            if payload_len != PAYLOAD_SIZE:
                self.get_logger().warn(
                    f"Unexpected payload length {payload_len}, expected {PAYLOAD_SIZE}."
                    " Re-syncing."
                )
                self.frames_bad_ += 1
                return

            # Read payload
            payload = self.serial_.read(payload_len)
            if len(payload) < payload_len:
                self.get_logger().warn("Incomplete payload read — skipping frame.")
                self.frames_bad_ += 1
                return

            # Verify CRC
            # header_crc_input = the first 4 bytes the CRC covers:
            #   0x5A, 0xA5, LEN_L, LEN_H
            header_crc_input = bytes(
                [FRAME_HEADER_1, FRAME_HEADER_2, rest_header[0], rest_header[1]]
            )
            computed_crc = compute_frame_crc(header_crc_input, payload)
            if computed_crc != frame_crc:
                self.get_logger().warn(
                    f"CRC mismatch: computed 0x{computed_crc:04X}, "
                    f"got 0x{frame_crc:04X} — dropping frame."
                )
                self.frames_bad_ += 1
                return

            # Verify packet label
            if payload[0] != PAYLOAD_LABEL:
                self.get_logger().warn(f"Unexpected packet label 0x{payload[0]:02X}.")
                self.frames_bad_ += 1
                return

            # Parse and publish
            data = parse_payload(payload)
            now = self.get_clock().now().to_msg()
            self._publish_imu(data, now)
            self._publish_mag(data, now)
            self._publish_pressure(data, now)
            if self.pub_euler_:
                self._publish_euler(data, now)

            self.frames_ok_ += 1
            if self.frames_ok_ % 500 == 0:  # log every 500 good frames
                self.get_logger().info(
                    f"Frames OK: {self.frames_ok_}  Bad: {self.frames_bad_}"
                )

        except serial.SerialException as e:
            self.get_logger().error(f"Serial error: {e}")

    # --------------------------------------------------------------------------
    def _sync_to_frame_header(self) -> bool:
        """
        Scan the byte stream until we see [0x5A, 0xA5] (2-byte frame header).
        Returns True when positioned just after the header, False on timeout.
        """
        byte = self.serial_.read(1)
        if not byte or byte[0] != FRAME_HEADER_1:
            return False
        byte = self.serial_.read(1)
        if not byte or byte[0] != FRAME_HEADER_2:
            return False
        return True

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
        """Clean up the serial port on shutdown."""
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
