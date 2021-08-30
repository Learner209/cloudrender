import os
from glob import glob
from setuptools import setup, find_packages
version = '1.0.0'

with open("README.md", "r") as fi:
    long_description = fi.read()

keywords = ["rendering", "pointcloud", "opengl", "mesh"]

classifiers = [
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3'
    ]

requirements = [
        "numpy",
        "scipy",
        "pyglm>=2.2.0",
        "trimesh",
        "torch",
        "tqdm",
        "smplpytorch",
        "PyOpenGL==3.1.5",
        "videoio>=0.2.3"
]

# Include shaders
# for filepath in glob('cloudrender/render/shaders/**/*.glsl', recursive=True):
#     dirpath = os.path.dirname(filepath)
#     if dirpath not in data_files:
#         data_files[dirpath] = []
#     data_files[dirpath].append(filepath)
# data_files = [(k,v) for k,v in data_files.items()]
package_path = "cloudrender/render/shaders"
package_files = {"cloudrender.render.shaders": [os.path.relpath(package_path, x)
                                                for x in glob('cloudrender/render/shaders/**/*.glsl', recursive=True)]}

setup(
    name="cloudrender",
    packages=find_packages(),
    package_data=package_files,
    version=version,
    description="An OpenGL framework for pointcloud and mesh rendering",
    author="Vladimir Guzov",
    author_email="vguzov@mpi-inf.mpg.de",
    url="https://github.com/vguzov/cloudrender",
    keywords=keywords,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    classifiers=classifiers
)