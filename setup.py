import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setuptools.setup(
    name='beer-game-environment',
    version='0.1',
    license='MIT',
    author='Filip Komljenovic',
    author_email='filip.komljenovic0@gmail.com',
    description=long_description,
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=required,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7'
)
