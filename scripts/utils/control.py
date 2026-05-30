# utils/control.py
from geometry_msgs.msg import Twist

# Tunable speeds
# LINEAR_FWD = 0.18
# ANGULAR_TURN = 0.6
# LINEAR_BIT = 0.08
# ANGULAR_BIT = 0.35

LINEAR_FWD = 0.10
ANGULAR_TURN = 0.40
LINEAR_BIT = 0.06
ANGULAR_BIT = 0.20

def make_twist(v, w):
    t = Twist()
    t.linear.x = float(v)
    t.angular.z = float(w)
    return t

def do_action(pub, action):
    """
    action: integer
      0 = forward
      1 = turn left (wide)
      2 = turn right (wide)
      3 = left bit (small turn)
      4 = right bit (small turn)
    pub: ROS publisher for Twist
    """
    if action == 0:
        pub.publish(make_twist(LINEAR_FWD, 0.0))
    elif action == 1:
        pub.publish(make_twist(0.08, ANGULAR_TURN))
    elif action == 2:
        pub.publish(make_twist(0.08, -ANGULAR_TURN))
    elif action == 3:
        pub.publish(make_twist(LINEAR_BIT, ANGULAR_BIT))
    elif action == 4:
        pub.publish(make_twist(LINEAR_BIT, -ANGULAR_BIT))
    else:
        # safety: stop
        pub.publish(make_twist(0.0, 0.0))

def stop(pub):
    pub.publish(make_twist(0.0, 0.0))
