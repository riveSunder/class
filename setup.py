from setuptools import setup 

setup(name="class",
    packages=["dw", "srnca", "rsvr"],
    version="0.0.2025.8",
    description="Supporting materials for CLASS",
    install_requires=["numpy==2.3.2",
        "matplotlib==3.10.5",
        "torch==2.7.1",
        "gymnasium==1.2.0",
        "mujoco==3.3.5",
        "torchvision==0.22.1",
        "scikit-image==0.25.2",
        "notebook==7.4.5",
        "jupyter==1.1.1"],
    )


    
