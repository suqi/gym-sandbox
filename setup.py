from setuptools import setup, find_packages

setup(
    name='gym_sandbox',
    version='0.1.0',
    description='Gym Env for MultiAgent Games',
    url='https://github.com/suqi/gym-multiagent',
    author='Matt Su',
    author_email='deqi.su@gmail.com',
    license='MIT License',
    packages=['gym_sandbox'],  #find_packages(),
    zip_safe=False,
    install_requires=['gym>=0.8.0', 'bokeh>=0.12'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
