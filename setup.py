from setuptools import setup, find_packages

setup(
    name='jenga',
    version='0.1',
    package_dir={'': 'src'},               # 告诉 setuptools 去 src/ 下找包
    packages=find_packages(where='src'),   # 自动找到 src 下的 jenga 包
)
