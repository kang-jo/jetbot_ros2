import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir, LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

from ament_index_python.packages import get_package_share_directory
 
def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
     
    robot_name = DeclareLaunchArgument('robot_name', default_value='jetbotV21')
    robot_model = DeclareLaunchArgument('robot_model', default_value='jetbotV21')  # jetbot_ros
    
    robot_x = DeclareLaunchArgument('x', default_value='-0.016799')
    robot_y = DeclareLaunchArgument('y', default_value='0.014966')
    robot_z = DeclareLaunchArgument('z', default_value='-0.043600')
    robot_roll = DeclareLaunchArgument('roll', default_value='-0.000015')
    robot_pitch = DeclareLaunchArgument('pitch', default_value='-0.026202')
    robot_yaw = DeclareLaunchArgument('yaw', default_value='-3.118884')


    world_file_name = 'test2.world'
    pkg_dir = get_package_share_directory('jetbot_ros')
 
    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')
 
    world = os.path.join(pkg_dir, 'worlds', world_file_name)
    launch_file_dir = os.path.join(pkg_dir, 'launch')
 
    gazebo = ExecuteProcess(
                cmd=['gazebo', '--verbose', world, 
                     '-s', 'libgazebo_ros_init.so', 
                     '-s', 'libgazebo_ros_factory.so'],
                output='screen', emulate_tty=True)

    
    spawn_entity = Node(package='jetbot_ros', executable='gazebo_spawn',   # FYI 'node_executable' is renamed to 'executable' in Foxy
                        parameters=[
                            {'name': LaunchConfiguration('robot_name')},
                            {'model': LaunchConfiguration('robot_model')},
                            {'x': LaunchConfiguration('x')},
                            {'y': LaunchConfiguration('y')},
                            {'z': LaunchConfiguration('z')},
                            {'roll': LaunchConfiguration('roll')},
                            {'pitch': LaunchConfiguration('pitch')},
                            {'yaw': LaunchConfiguration('yaw')},
                        ],
                        output='screen', emulate_tty=True)
 
    rviz_config = os.path.join(pkg_dir, 'rviz', 'default.rviz')
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config])

    return LaunchDescription([
        robot_name,
        robot_model,
        robot_x,
        robot_y,
        robot_z,
        robot_roll,
        robot_pitch,
        robot_yaw,
        gazebo,
        spawn_entity,
    ])
