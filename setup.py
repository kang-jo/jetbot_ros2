from setuptools import setup, find_packages
from glob import glob
from itertools import chain

import os

package_name = 'jetbot_ros'

def generate_data_files(dirs=['launch', 'gazebo/worlds', 'gazebo/models']):
    """
    Generate recursive list of data files, without listing directories in the output.
    """
    data_files = []
    for path, _, files in chain.from_iterable(os.walk(dir) for dir in dirs):
        install_dir = path[len('gazebo/'):] if path.startswith('gazebo/') else path  # remove gazebo/ prefix
        install_dir = os.path.join('share', package_name, install_dir)
        list_entry = (install_dir, [os.path.join(path, f) for f in files if not f.startswith('.')])
        data_files.append(list_entry)
    return data_files
    
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + generate_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Devan Al Fauzi',
    maintainer_email='devan@nvidia.com',
    description='ROS nodes for JetBot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gazebo_spawn = jetbot_ros.gazebo_spawn:main',
            'motors_nvidia = jetbot_ros.motors_nvidia:main',
            'motors_waveshare = jetbot_ros.motors_waveshare:main',
            'motors_sparkfun = jetbot_ros.motors_sparkfun:main',
            'oled_ssd1306 = jetbot_ros.oled_ssd1306:main',
        ],
    },
)
