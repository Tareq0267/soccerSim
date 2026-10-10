from glob import glob

from setuptools import setup

setup(
    name="soccer_sim_bridge",
    version="0.1.0",
    packages=["soccer_sim_bridge"],
    package_data={"soccer_sim_bridge": ["controller.html"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/soccer_sim_bridge"]),
        ("share/soccer_sim_bridge", ["package.xml"]),
        ("share/soccer_sim_bridge/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": ["sim_bridge = soccer_sim_bridge.bridge_node:main"]},
)
