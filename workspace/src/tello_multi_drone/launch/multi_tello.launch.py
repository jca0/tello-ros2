from launch import LaunchDescription
from launch_ros.actions import Node
import yaml

def generate_launch_description():
    with open("../config/drone_params.yaml", "r") as f:
        drone_params = yaml.safe_load(f)

    nodes = []

    for drone in drone_params['multi_drone_controller']['ros__parameters']['drones']:
        nodes.append(
            Node(
                package='tello',
                executable='tello',
                output='screen',
                name=drone['name'],
                parameters=[
                    {'connect_timeout': 10.0},
                    {'tello_ip': drone['ip']},
                    {'tf_base': 'map'},
                    {'tf_drone': drone['name']}
                ],
                remappings=[
                    ('/image_raw', '/camera')
                ],
                respawn=True
            ),

            # Tello control node
            Node(
                package='tello_control',
                executable='tello_control',
                namespace='/',
                name='control',
                output='screen',
                respawn=False
            ),

            # RQT topic debug tool
            Node(
                package='rqt_gui',
                executable='rqt_gui',
                output='screen',
                namespace='/',
                name='rqt',
                respawn=False
            ),

            # RViz data visualization tool
            Node(
                package='rviz2',
                executable='rviz2',
                output='screen',
                namespace='/',
                name='rviz2',
                respawn=True,
                arguments=['-d', '/home/tentone/Git/tello-slam/workspace/src/rviz.rviz']
            ),

            # Static TF publisher
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                namespace='/',
                name='tf',
                arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'drone'],
                respawn=True
            ),
        )

    return LaunchDescription(nodes)