from setuptools import setup, find_packages

setup(
    name="beets-replaygainffmpeg",
    version="0.0.1",
    description="beets plugin to use ffmpeg to do replaygain (copied from git beats)",
    author='Timothy Redaelli',
    author_email='timothy.redaelli@gmail.com',
    url='https://github.com/drizzt/beets-replaygainffmpeg',
    download_url='https://github.com/drizzt/beets-replaygainffmpeg.git',
    license='MIT',
    platforms='ALL',

    packages=['beetsplug'],
    namespace_packages=['beetsplug'],
    install_requires=['beets>=1.3.11','beets<1.5.0'],

    classifiers=[
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Multimedia :: Sound/Audio :: Players :: MP3',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ]
)
