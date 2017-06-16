from setuptools import setup, find_packages

setup(
    name='gym_multiagent',
    version='0.1.0',
    description='Gym Env for MultiAgent Games',
    url='https://github.com/suqi/gym-multiagent',
    author='Matt Su',
    author_email='deqi.su@gmail.com',
    license='MIT License',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['gym>=0.8.0', 'bokeh>=0.12'],
)
