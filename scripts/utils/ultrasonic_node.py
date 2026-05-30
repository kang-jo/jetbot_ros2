import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import serial

class UltrasonicNode(Node):
    def __init__(self):
        super().__init__('ultrasonic_node')

        # Serial config
        self.ser = serial.Serial('/dev/ttyTHS1', 9600, timeout=1)

        # Publisher masing-masing sensor
        self.pubs = {
            'left2': self.create_publisher(Float32, '/jetbotV2/ultrasonic_left_2', 10),
            'left1': self.create_publisher(Float32, '/jetbotV2/ultrasonic_left_1', 10),
            'left0': self.create_publisher(Float32, '/jetbotV2/ultrasonic_left_0', 10),
            'front': self.create_publisher(Float32, '/jetbotV2/ultrasonic_front', 10),
            'right0': self.create_publisher(Float32, '/jetbotV2/ultrasonic_right_0', 10),
            'right1': self.create_publisher(Float32, '/jetbotV2/ultrasonic_right_1', 10),
            'right2': self.create_publisher(Float32, '/jetbotV2/ultrasonic_right_2', 10),
        }

        # Urutan sesuai data kamu
        self.order = ['left2', 'left1', 'left0', 'front', 'right0', 'right1', 'right2']

        # Timer untuk baca serial
        self.timer = self.create_timer(0.05, self.read_serial)  # 20 Hz

    def read_serial(self):
        try:
            line = self.ser.readline().decode(errors='ignore').strip()

            if not line:
                return

            parts = line.split(',')

            # 🔴 Filter data tidak lengkap
            if len(parts) != 7:
                self.get_logger().warn(f"Data tidak lengkap: {line}")
                return

            try:
                values = [float(p) for p in parts]
            except ValueError:
                self.get_logger().warn(f"Data invalid: {line}")
                return

            # Publish ke masing-masing topic
            for key, value in zip(self.order, values):
                msg = Float32()
                msg.data = value
                self.pubs[key].publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()