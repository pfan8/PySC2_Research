# PySC2调研
## 1. 环境搭建
### Hardware+Linux
本次调研使用的单机，在拓展的SSD上安装了Ubuntu 16.04 LTS，在该系统下进行环境搭建，硬件配置大致如下:
	
![](https://image.ibb.co/k0KwEH/Screenshot_from_2018_04_30_14_06_11.png) 

台机是2015年自行组装的，CPU使用的是E3-1231 v3，8核 主频3.4GHz，和当时的i7配置相当，并且应当适合运行多线程任务，然而在<span style="color:red">pysc2-examples</span>中，跑A2C算法进行训练时效果不理想，由于github上的[项目](https://github.com/chris-chris/pysc2-examples)更新过了，也有可能是代码就有问题，在issue里看到了同样的问题，目前还没有人解答（P.S. 没有查看GPU的使用情况，应该就是tensorflow的CPU mode）
### PySC2
编译器使用的是Ubuntu自带的python 3.5
``` sh
pfan8@pfan8-MS-7816:~$ python3 -V
Python 3.5.2
pfan8@pfan8-MS-7816:~$ python3 -m pip -V
pip 10.0.1 from /home/pfan8/.local/lib/python3.5/site-packages/pip (python 3.5)
```
接下来就是按照[pysc2项目](https://github.com/deepmind/pysc2)的教程进行搭建：
1. 通过pip过着git安装pysc2到python3
2. 安装linux下的[SCII游戏](https://github.com/Blizzard/s2client-proto#downloads)，这里建议安装3.16.1版本的，PySC2目前只支持到3.19，4.0.2版本的不支持replay功能，我尝试更改

	pysc2/pysc2/run_configs/platforms.py
文件，添加新版本绕过check，不过程序依然会运行失败
3. 添加ladder maps与mini games到游戏目录（默认~/StarCraftII/Maps），<span style="color:red">**mini games需要在PySC2中下载，并且Maps目录下需要保留子文件夹，和Win平台的C++项目不同**</span>，即目录结构应如下（只列出目录）：
```
.
├── AppData
│   └── Maps
│       └── Cache
│           └── TempLaunchMap.SC2Map
│               └── 00000000
├── Battle.net
├── Libs
├── Maps
│   ├── Ladder2017Season1
│   ├── Ladder2017Season2
│   ├── Ladder2017Season3
│   ├── Ladder2017Season4
│   ├── Ladder2018Season1
│   ├── Melee
│   └── mini_games
├── Replays
│   ├── local
│   ├── RandomAgent
│   └── SimpleAgent
├── SC2Data
│   ├── config
│   │   ├── 1f
│   │   │   └── 3b
│   │   ├── 32
│   │   │   └── 5e
│   │   └── eb
│   │       └── a0
│   ├── data
│   ├── indices
│   └── patch
└── Versions
    └── Base55958
```
接下来便可以在PySC2上运行Bot了
### pysc2-examples
从git上down下来最新的代码运行，会出错，报错如下

> AttributeError: module 'baselines.common.tf_util' has no attribute 'BatchInput'

需要替换U.BatchInput为deepq.utils.BatchInput；
以及替换U.load_state为deepq.utils.load_state
安装baseline的时候出过问题，有一些依赖软件要自行下载，如cmake等，导致我不确定baseline是否安装成功了，能用，但是用pysc2-tutorial1里面的train_mineral_shards.py里的a2c算法训练模型时出现reward不断下降的状态，我用自己的单机（CPU:E3-1231, Mem:8G，GPU:AMD R9-390）跑了一夜，发现reward就一直在40到60之间徘徊,而程序一直未终止训练。

<div id="" style="overflow:scroll; height:300px; white-space: pre;"> 
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.0
mean_100ep_reward_a2c 48.3
mean_100ep_reward_a2c 48.5
mean_100ep_reward_a2c 48.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.8
mean_100ep_reward_a2c 49.1
mean_100ep_reward_a2c 49.2
mean_100ep_reward_a2c 49.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.3
mean_100ep_reward_a2c 49.2
mean_100ep_reward_a2c 49.5
mean_100ep_reward_a2c 49.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.5
mean_100ep_reward_a2c 49.3
mean_100ep_reward_a2c 49.6
mean_100ep_reward_a2c 49.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.7
mean_100ep_reward_a2c 50.0
mean_100ep_reward_a2c 50.1
mean_100ep_reward_a2c 50.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.5
mean_100ep_reward_a2c 50.6
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.8
mean_100ep_reward_a2c 54.0
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 54.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
mean_100ep_reward_a2c 54.7
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
mean_100ep_reward_a2c 55.6
mean_100ep_reward_a2c 55.8
mean_100ep_reward_a2c 56.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.3
mean_100ep_reward_a2c 56.5
mean_100ep_reward_a2c 56.6
mean_100ep_reward_a2c 56.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.0
mean_100ep_reward_a2c 57.0
mean_100ep_reward_a2c 57.1
mean_100ep_reward_a2c 57.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.4
mean_100ep_reward_a2c 57.5
mean_100ep_reward_a2c 57.4
mean_100ep_reward_a2c 57.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.8
mean_100ep_reward_a2c 57.8
mean_100ep_reward_a2c 57.9
mean_100ep_reward_a2c 57.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.0
mean_100ep_reward_a2c 57.9
mean_100ep_reward_a2c 57.9
mean_100ep_reward_a2c 58.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.2
mean_100ep_reward_a2c 58.4
mean_100ep_reward_a2c 58.3
mean_100ep_reward_a2c 58.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.2
mean_100ep_reward_a2c 58.2
mean_100ep_reward_a2c 58.3
mean_100ep_reward_a2c 58.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.1
mean_100ep_reward_a2c 58.1
mean_100ep_reward_a2c 58.1
mean_100ep_reward_a2c 58.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.1
mean_100ep_reward_a2c 58.2
mean_100ep_reward_a2c 58.2
mean_100ep_reward_a2c 58.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.3
mean_100ep_reward_a2c 58.6
mean_100ep_reward_a2c 58.5
mean_100ep_reward_a2c 58.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.7
mean_100ep_reward_a2c 58.6
mean_100ep_reward_a2c 58.7
mean_100ep_reward_a2c 58.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.9
mean_100ep_reward_a2c 58.9
mean_100ep_reward_a2c 59.1
mean_100ep_reward_a2c 59.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.8
mean_100ep_reward_a2c 58.8
mean_100ep_reward_a2c 58.7
mean_100ep_reward_a2c 58.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.5
mean_100ep_reward_a2c 58.6
mean_100ep_reward_a2c 58.6
mean_100ep_reward_a2c 58.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.5
mean_100ep_reward_a2c 58.7
mean_100ep_reward_a2c 58.7
mean_100ep_reward_a2c 58.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.8
mean_100ep_reward_a2c 58.9
mean_100ep_reward_a2c 59.0
mean_100ep_reward_a2c 59.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.0
mean_100ep_reward_a2c 58.9
mean_100ep_reward_a2c 58.7
mean_100ep_reward_a2c 58.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.9
mean_100ep_reward_a2c 59.0
mean_100ep_reward_a2c 59.0
mean_100ep_reward_a2c 58.8
init group list
init group list
Game has started.
mean_100ep_reward_a2c 58.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.1
mean_100ep_reward_a2c 59.1
mean_100ep_reward_a2c 59.0
Game has started.
mean_100ep_reward_a2c 59.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.9
mean_100ep_reward_a2c 58.9
mean_100ep_reward_a2c 59.0
Game has started.
mean_100ep_reward_a2c 59.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.0
mean_100ep_reward_a2c 59.0
mean_100ep_reward_a2c 59.0
Game has started.
mean_100ep_reward_a2c 59.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.1
mean_100ep_reward_a2c 59.2
mean_100ep_reward_a2c 59.5
Game has started.
mean_100ep_reward_a2c 59.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.6
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 60.0
Game has started.
mean_100ep_reward_a2c 60.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 60.1
mean_100ep_reward_a2c 60.0
Game has started.
mean_100ep_reward_a2c 60.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 60.0
Game has started.
mean_100ep_reward_a2c 59.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.7
mean_100ep_reward_a2c 59.5
mean_100ep_reward_a2c 59.5
Game has started.
mean_100ep_reward_a2c 59.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.2
mean_100ep_reward_a2c 59.3
mean_100ep_reward_a2c 59.3
Game has started.
mean_100ep_reward_a2c 59.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.1
mean_100ep_reward_a2c 59.1
mean_100ep_reward_a2c 59.2
Game has started.
mean_100ep_reward_a2c 59.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.2
mean_100ep_reward_a2c 59.1
mean_100ep_reward_a2c 59.1
Game has started.
mean_100ep_reward_a2c 59.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.1
mean_100ep_reward_a2c 59.3
mean_100ep_reward_a2c 59.2
Game has started.
mean_100ep_reward_a2c 59.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.0
mean_100ep_reward_a2c 58.9
mean_100ep_reward_a2c 59.1
Game has started.
mean_100ep_reward_a2c 59.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.4
mean_100ep_reward_a2c 59.4
mean_100ep_reward_a2c 59.4
Game has started.
mean_100ep_reward_a2c 59.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.6
mean_100ep_reward_a2c 59.5
mean_100ep_reward_a2c 59.5
Game has started.
mean_100ep_reward_a2c 59.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.5
mean_100ep_reward_a2c 59.4
mean_100ep_reward_a2c 59.5
Game has started.
mean_100ep_reward_a2c 59.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.6
mean_100ep_reward_a2c 59.6
mean_100ep_reward_a2c 59.7
Game has started.
mean_100ep_reward_a2c 59.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 60.1
mean_100ep_reward_a2c 60.0
Game has started.
mean_100ep_reward_a2c 60.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.8
mean_100ep_reward_a2c 59.8
mean_100ep_reward_a2c 59.8
Game has started.
mean_100ep_reward_a2c 59.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.9
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 60.0
Game has started.
mean_100ep_reward_a2c 60.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.9
mean_100ep_reward_a2c 59.9
mean_100ep_reward_a2c 60.1
Game has started.
mean_100ep_reward_a2c 60.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.1
mean_100ep_reward_a2c 60.1
mean_100ep_reward_a2c 60.2
Game has started.
mean_100ep_reward_a2c 60.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.3
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 59.8
Game has started.
mean_100ep_reward_a2c 59.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 60.2
mean_100ep_reward_a2c 60.5
Game has started.
mean_100ep_reward_a2c 60.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.3
mean_100ep_reward_a2c 60.2
mean_100ep_reward_a2c 60.3
Game has started.
mean_100ep_reward_a2c 60.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.3
mean_100ep_reward_a2c 60.1
mean_100ep_reward_a2c 60.1
Game has started.
mean_100ep_reward_a2c 60.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.1
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 59.9
Game has started.
mean_100ep_reward_a2c 59.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.7
mean_100ep_reward_a2c 59.8
mean_100ep_reward_a2c 59.8
Game has started.
mean_100ep_reward_a2c 59.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 59.9
init group list
init group list
init group list
Game has started.
Game has started.
mean_100ep_reward_a2c 59.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.9
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 59.7
Game has started.
Game has started.
mean_100ep_reward_a2c 59.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.7
mean_100ep_reward_a2c 59.9
mean_100ep_reward_a2c 59.9
Game has started.
Game has started.
mean_100ep_reward_a2c 60.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.1
mean_100ep_reward_a2c 60.0
mean_100ep_reward_a2c 60.1
Game has started.
Game has started.
mean_100ep_reward_a2c 60.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.1
mean_100ep_reward_a2c 60.3
mean_100ep_reward_a2c 60.4
Game has started.
Game has started.
mean_100ep_reward_a2c 60.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.7
mean_100ep_reward_a2c 60.7
Game has started.
Game has started.
mean_100ep_reward_a2c 60.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.6
Game has started.
Game has started.
mean_100ep_reward_a2c 60.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.7
mean_100ep_reward_a2c 60.7
mean_100ep_reward_a2c 60.9
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
mean_100ep_reward_a2c 60.9
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.9
mean_100ep_reward_a2c 61.0
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.0
mean_100ep_reward_a2c 60.9
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.8
mean_100ep_reward_a2c 60.8
mean_100ep_reward_a2c 60.8
Game has started.
Game has started.
mean_100ep_reward_a2c 60.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.7
Game has started.
Game has started.
mean_100ep_reward_a2c 60.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.0
mean_100ep_reward_a2c 61.1
mean_100ep_reward_a2c 61.3
Game has started.
Game has started.
mean_100ep_reward_a2c 61.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.2
mean_100ep_reward_a2c 61.4
mean_100ep_reward_a2c 61.2
Game has started.
Game has started.
mean_100ep_reward_a2c 61.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.2
mean_100ep_reward_a2c 60.9
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
mean_100ep_reward_a2c 61.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.1
mean_100ep_reward_a2c 61.2
mean_100ep_reward_a2c 61.2
Game has started.
Game has started.
mean_100ep_reward_a2c 61.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.2
mean_100ep_reward_a2c 61.2
mean_100ep_reward_a2c 61.1
Game has started.
Game has started.
mean_100ep_reward_a2c 61.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.9
mean_100ep_reward_a2c 61.0
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
mean_100ep_reward_a2c 60.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.5
Game has started.
Game has started.
mean_100ep_reward_a2c 60.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.7
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.7
Game has started.
Game has started.
mean_100ep_reward_a2c 60.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.4
Game has started.
Game has started.
mean_100ep_reward_a2c 60.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.6
mean_100ep_reward_a2c 60.8
mean_100ep_reward_a2c 60.8
Game has started.
Game has started.
mean_100ep_reward_a2c 60.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.9
mean_100ep_reward_a2c 60.9
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
mean_100ep_reward_a2c 61.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.3
mean_100ep_reward_a2c 61.4
mean_100ep_reward_a2c 61.5
Game has started.
Game has started.
mean_100ep_reward_a2c 61.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.5
mean_100ep_reward_a2c 61.5
mean_100ep_reward_a2c 61.5
Game has started.
Game has started.
mean_100ep_reward_a2c 61.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.4
mean_100ep_reward_a2c 61.3
mean_100ep_reward_a2c 61.3
Game has started.
Game has started.
mean_100ep_reward_a2c 61.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.1
mean_100ep_reward_a2c 61.1
mean_100ep_reward_a2c 60.9
Game has started.
Game has started.
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.0
mean_100ep_reward_a2c 60.9
mean_100ep_reward_a2c 60.8
Game has started.
Game has started.
mean_100ep_reward_a2c 60.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.0
mean_100ep_reward_a2c 61.0
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
mean_100ep_reward_a2c 60.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.1
mean_100ep_reward_a2c 61.1
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 61.0
mean_100ep_reward_a2c 61.2
mean_100ep_reward_a2c 61.0
Game has started.
Game has started.
mean_100ep_reward_a2c 60.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 60.5
mean_100ep_reward_a2c 60.2
mean_100ep_reward_a2c 59.8
Game has started.
Game has started.
mean_100ep_reward_a2c 59.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.7
mean_100ep_reward_a2c 59.7
mean_100ep_reward_a2c 59.5
Game has started.
Game has started.
mean_100ep_reward_a2c 59.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.3
mean_100ep_reward_a2c 59.2
mean_100ep_reward_a2c 58.8
Game has started.
Game has started.
mean_100ep_reward_a2c 58.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.7
mean_100ep_reward_a2c 58.7
mean_100ep_reward_a2c 58.8
Game has started.
Game has started.
mean_100ep_reward_a2c 58.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.5
mean_100ep_reward_a2c 58.6
mean_100ep_reward_a2c 58.5
Game has started.
Game has started.
mean_100ep_reward_a2c 58.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.7
mean_100ep_reward_a2c 58.6
mean_100ep_reward_a2c 58.6
Game has started.
Game has started.
mean_100ep_reward_a2c 58.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.0
mean_100ep_reward_a2c 57.8
mean_100ep_reward_a2c 57.6
Game has started.
Game has started.
mean_100ep_reward_a2c 57.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.3
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.0
Game has started.
Game has started.
mean_100ep_reward_a2c 57.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.8
mean_100ep_reward_a2c 56.6
mean_100ep_reward_a2c 56.6
Game has started.
Game has started.
mean_100ep_reward_a2c 56.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.6
mean_100ep_reward_a2c 56.6
mean_100ep_reward_a2c 56.5
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
mean_100ep_reward_a2c 56.6
mean_100ep_reward_a2c 56.6
Game has started.
Game has started.
mean_100ep_reward_a2c 56.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.5
mean_100ep_reward_a2c 56.7
mean_100ep_reward_a2c 56.6
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
mean_100ep_reward_a2c 56.7
mean_100ep_reward_a2c 56.7
Game has started.
Game has started.
mean_100ep_reward_a2c 56.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.8
mean_100ep_reward_a2c 56.6
mean_100ep_reward_a2c 56.6
Game has started.
Game has started.
mean_100ep_reward_a2c 56.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.4
mean_100ep_reward_a2c 56.3
mean_100ep_reward_a2c 56.2
Game has started.
Game has started.
mean_100ep_reward_a2c 56.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.1
mean_100ep_reward_a2c 56.1
mean_100ep_reward_a2c 56.0
Game has started.
Game has started.
mean_100ep_reward_a2c 56.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.8
mean_100ep_reward_a2c 55.8
mean_100ep_reward_a2c 55.7
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 54.7
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.5
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 53.5
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
mean_100ep_reward_a2c 53.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 53.8
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.2
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.2
Game has started.
Game has started.
mean_100ep_reward_a2c 54.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.4
Game has started.
Game has started.
mean_100ep_reward_a2c 54.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
mean_100ep_reward_a2c 54.3
mean_100ep_reward_a2c 54.4
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
mean_100ep_reward_a2c 54.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.5
Game has started.
Game has started.
mean_100ep_reward_a2c 54.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.6
mean_100ep_reward_a2c 54.5
mean_100ep_reward_a2c 54.4
Game has started.
Game has started.
mean_100ep_reward_a2c 54.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.3
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
mean_100ep_reward_a2c 54.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.5
mean_100ep_reward_a2c 54.4
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.3
mean_100ep_reward_a2c 54.4
Game has started.
Game has started.
mean_100ep_reward_a2c 54.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.7
mean_100ep_reward_a2c 55.8
mean_100ep_reward_a2c 55.7
Game has started.
Game has started.
mean_100ep_reward_a2c 55.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.9
mean_100ep_reward_a2c 55.7
mean_100ep_reward_a2c 55.7
Game has started.
Game has started.
mean_100ep_reward_a2c 55.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.9
mean_100ep_reward_a2c 56.0
mean_100ep_reward_a2c 56.0
Game has started.
Game has started.
mean_100ep_reward_a2c 56.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.1
mean_100ep_reward_a2c 56.2
mean_100ep_reward_a2c 56.3
Game has started.
Game has started.
mean_100ep_reward_a2c 56.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.5
mean_100ep_reward_a2c 56.5
mean_100ep_reward_a2c 56.5
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
mean_100ep_reward_a2c 56.5
mean_100ep_reward_a2c 56.5
Game has started.
Game has started.
mean_100ep_reward_a2c 56.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.3
mean_100ep_reward_a2c 56.5
mean_100ep_reward_a2c 56.5
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.9
mean_100ep_reward_a2c 56.9
mean_100ep_reward_a2c 57.1
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.1
mean_100ep_reward_a2c 57.0
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.3
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.4
Game has started.
Game has started.
mean_100ep_reward_a2c 57.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.5
mean_100ep_reward_a2c 57.7
mean_100ep_reward_a2c 57.6
Game has started.
Game has started.
mean_100ep_reward_a2c 57.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.6
mean_100ep_reward_a2c 57.6
mean_100ep_reward_a2c 57.6
Game has started.
Game has started.
mean_100ep_reward_a2c 57.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.6
mean_100ep_reward_a2c 57.6
mean_100ep_reward_a2c 57.8
Game has started.
Game has started.
mean_100ep_reward_a2c 57.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.8
mean_100ep_reward_a2c 57.7
mean_100ep_reward_a2c 57.6
Game has started.
Game has started.
mean_100ep_reward_a2c 57.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.6
mean_100ep_reward_a2c 57.5
mean_100ep_reward_a2c 57.4
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.1
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.3
Game has started.
Game has started.
mean_100ep_reward_a2c 57.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.4
mean_100ep_reward_a2c 57.4
mean_100ep_reward_a2c 57.6
Game has started.
Game has started.
mean_100ep_reward_a2c 57.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.9
mean_100ep_reward_a2c 57.8
mean_100ep_reward_a2c 57.9
Game has started.
Game has started.
mean_100ep_reward_a2c 58.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.3
mean_100ep_reward_a2c 58.3
mean_100ep_reward_a2c 58.3
Game has started.
Game has started.
mean_100ep_reward_a2c 58.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.2
mean_100ep_reward_a2c 58.0
mean_100ep_reward_a2c 57.8
Game has started.
Game has started.
mean_100ep_reward_a2c 57.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.7
mean_100ep_reward_a2c 57.8
mean_100ep_reward_a2c 57.8
Game has started.
Game has started.
mean_100ep_reward_a2c 57.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.6
mean_100ep_reward_a2c 57.4
mean_100ep_reward_a2c 56.9
Game has started.
Game has started.
mean_100ep_reward_a2c 56.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.2
mean_100ep_reward_a2c 56.0
mean_100ep_reward_a2c 55.8
Game has started.
Game has started.
mean_100ep_reward_a2c 55.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.7
mean_100ep_reward_a2c 55.8
mean_100ep_reward_a2c 55.8
Game has started.
Game has started.
mean_100ep_reward_a2c 55.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.8
mean_100ep_reward_a2c 55.7
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
mean_100ep_reward_a2c 54.6
mean_100ep_reward_a2c 54.4
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 53.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
mean_100ep_reward_a2c 54.0
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.0
mean_100ep_reward_a2c 53.9
Game has started.
Game has started.
mean_100ep_reward_a2c 53.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 53.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.8
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 53.8
Game has started.
Game has started.
mean_100ep_reward_a2c 53.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.6
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 52.2
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.4
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 55.1
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.7
mean_100ep_reward_a2c 55.8
mean_100ep_reward_a2c 55.9
Game has started.
Game has started.
mean_100ep_reward_a2c 56.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.2
mean_100ep_reward_a2c 56.3
mean_100ep_reward_a2c 56.3
Game has started.
Game has started.
mean_100ep_reward_a2c 56.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.3
mean_100ep_reward_a2c 56.5
mean_100ep_reward_a2c 56.6
Game has started.
Game has started.
mean_100ep_reward_a2c 56.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
mean_100ep_reward_a2c 56.8
mean_100ep_reward_a2c 56.8
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.3
mean_100ep_reward_a2c 57.3
mean_100ep_reward_a2c 57.2
Game has started.
Game has started.
mean_100ep_reward_a2c 57.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.0
mean_100ep_reward_a2c 57.0
mean_100ep_reward_a2c 56.9
Game has started.
Game has started.
mean_100ep_reward_a2c 57.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.1
mean_100ep_reward_a2c 56.9
mean_100ep_reward_a2c 56.9
Game has started.
Game has started.
mean_100ep_reward_a2c 56.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.8
mean_100ep_reward_a2c 56.7
mean_100ep_reward_a2c 56.7
Game has started.
Game has started.
mean_100ep_reward_a2c 56.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.2
mean_100ep_reward_a2c 56.0
mean_100ep_reward_a2c 56.1
Game has started.
Game has started.
mean_100ep_reward_a2c 56.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.1
mean_100ep_reward_a2c 56.3
mean_100ep_reward_a2c 56.4
Game has started.
Game has started.
mean_100ep_reward_a2c 56.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.6
mean_100ep_reward_a2c 56.9
mean_100ep_reward_a2c 57.1
Game has started.
Game has started.
mean_100ep_reward_a2c 56.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.6
mean_100ep_reward_a2c 56.4
mean_100ep_reward_a2c 56.0
Game has started.
Game has started.
mean_100ep_reward_a2c 55.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 54.9
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
mean_100ep_reward_a2c 55.1
Max game loops reached. Current: 263176 Max: 262144
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Game has started.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
mean_100ep_reward_a2c 54.8
mean_100ep_reward_a2c 54.8
Game has started.
Game has started.
mean_100ep_reward_a2c 54.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.7
mean_100ep_reward_a2c 54.6
mean_100ep_reward_a2c 54.8
Game has started.
Game has started.
mean_100ep_reward_a2c 54.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.5
mean_100ep_reward_a2c 54.5
mean_100ep_reward_a2c 54.5
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.2
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.0
mean_100ep_reward_a2c 53.9
Game has started.
Game has started.
mean_100ep_reward_a2c 53.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 53.8
Game has started.
Game has started.
mean_100ep_reward_a2c 53.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.5
mean_100ep_reward_a2c 53.5
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.8
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.5
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.7
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.4
mean_100ep_reward_a2c 50.2
mean_100ep_reward_a2c 49.9
Game has started.
Game has started.
mean_100ep_reward_a2c 49.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.0
mean_100ep_reward_a2c 48.9
mean_100ep_reward_a2c 48.7
Game has started.
Game has started.
mean_100ep_reward_a2c 48.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.0
mean_100ep_reward_a2c 48.0
mean_100ep_reward_a2c 47.8
Game has started.
Game has started.
mean_100ep_reward_a2c 47.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.5
mean_100ep_reward_a2c 47.5
mean_100ep_reward_a2c 47.3
Game has started.
Game has started.
mean_100ep_reward_a2c 47.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.2
mean_100ep_reward_a2c 47.2
mean_100ep_reward_a2c 47.0
Game has started.
Game has started.
mean_100ep_reward_a2c 46.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.7
mean_100ep_reward_a2c 46.3
mean_100ep_reward_a2c 45.8
Game has started.
Game has started.
mean_100ep_reward_a2c 45.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.5
mean_100ep_reward_a2c 45.3
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.2
mean_100ep_reward_a2c 45.0
mean_100ep_reward_a2c 44.9
Game has started.
Game has started.
mean_100ep_reward_a2c 45.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.8
mean_100ep_reward_a2c 44.9
mean_100ep_reward_a2c 44.8
Game has started.
Game has started.
mean_100ep_reward_a2c 44.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.8
mean_100ep_reward_a2c 44.7
mean_100ep_reward_a2c 44.8
Game has started.
Game has started.
mean_100ep_reward_a2c 44.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.7
mean_100ep_reward_a2c 44.7
mean_100ep_reward_a2c 44.8
Game has started.
Game has started.
mean_100ep_reward_a2c 44.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.3
mean_100ep_reward_a2c 44.1
mean_100ep_reward_a2c 44.0
Game has started.
Game has started.
mean_100ep_reward_a2c 44.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 43.9
mean_100ep_reward_a2c 43.8
mean_100ep_reward_a2c 43.7
Game has started.
Game has started.
mean_100ep_reward_a2c 43.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 43.8
mean_100ep_reward_a2c 43.9
mean_100ep_reward_a2c 44.0
Game has started.
Game has started.
mean_100ep_reward_a2c 44.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.4
mean_100ep_reward_a2c 44.5
mean_100ep_reward_a2c 44.7
Game has started.
Game has started.
mean_100ep_reward_a2c 44.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.8
mean_100ep_reward_a2c 44.7
mean_100ep_reward_a2c 44.6
Game has started.
Game has started.
mean_100ep_reward_a2c 44.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.4
mean_100ep_reward_a2c 44.6
mean_100ep_reward_a2c 44.6
Game has started.
Game has started.
mean_100ep_reward_a2c 44.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.2
mean_100ep_reward_a2c 43.8
mean_100ep_reward_a2c 43.4
Game has started.
Game has started.
mean_100ep_reward_a2c 43.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 42.9
mean_100ep_reward_a2c 42.6
mean_100ep_reward_a2c 42.4
Game has started.
Game has started.
mean_100ep_reward_a2c 42.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 42.2
mean_100ep_reward_a2c 42.2
mean_100ep_reward_a2c 42.1
Game has started.
Game has started.
mean_100ep_reward_a2c 42.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 42.6
mean_100ep_reward_a2c 42.9
mean_100ep_reward_a2c 43.3
Game has started.
Game has started.
mean_100ep_reward_a2c 43.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 43.9
mean_100ep_reward_a2c 44.1
mean_100ep_reward_a2c 44.4
Game has started.
Game has started.
mean_100ep_reward_a2c 44.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.8
mean_100ep_reward_a2c 45.0
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
mean_100ep_reward_a2c 45.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.5
mean_100ep_reward_a2c 45.6
mean_100ep_reward_a2c 45.7
Game has started.
Game has started.
mean_100ep_reward_a2c 45.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.9
mean_100ep_reward_a2c 45.9
mean_100ep_reward_a2c 45.8
Game has started.
Game has started.
mean_100ep_reward_a2c 46.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.3
mean_100ep_reward_a2c 46.6
mean_100ep_reward_a2c 46.7
Game has started.
Game has started.
mean_100ep_reward_a2c 47.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.5
mean_100ep_reward_a2c 47.9
mean_100ep_reward_a2c 47.9
Game has started.
Game has started.
mean_100ep_reward_a2c 48.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.5
mean_100ep_reward_a2c 48.8
mean_100ep_reward_a2c 48.9
Game has started.
Game has started.
mean_100ep_reward_a2c 49.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.3
mean_100ep_reward_a2c 49.6
mean_100ep_reward_a2c 49.8
Game has started.
Game has started.
mean_100ep_reward_a2c 50.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.1
mean_100ep_reward_a2c 50.1
mean_100ep_reward_a2c 50.2
Game has started.
Game has started.
mean_100ep_reward_a2c 50.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.6
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.7
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
mean_100ep_reward_a2c 51.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.7
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
mean_100ep_reward_a2c 51.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.7
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 52.0
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.5
mean_100ep_reward_a2c 54.7
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.1
Game has started.
Game has started.
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
mean_100ep_reward_a2c 54.0
mean_100ep_reward_a2c 53.5
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.6
mean_100ep_reward_a2c 50.2
mean_100ep_reward_a2c 49.8
Game has started.
Game has started.
mean_100ep_reward_a2c 49.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.7
mean_100ep_reward_a2c 49.9
mean_100ep_reward_a2c 49.8
Game has started.
Game has started.
mean_100ep_reward_a2c 50.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.0
mean_100ep_reward_a2c 50.0
mean_100ep_reward_a2c 50.1
Game has started.
Game has started.
mean_100ep_reward_a2c 50.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.6
Game has started.
Game has started.
mean_100ep_reward_a2c 50.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.3
mean_100ep_reward_a2c 50.3
mean_100ep_reward_a2c 50.5
Game has started.
Game has started.
mean_100ep_reward_a2c 50.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.2
mean_100ep_reward_a2c 50.0
mean_100ep_reward_a2c 49.8
Game has started.
Game has started.
mean_100ep_reward_a2c 49.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.6
mean_100ep_reward_a2c 49.6
mean_100ep_reward_a2c 49.6
Game has started.
Game has started.
mean_100ep_reward_a2c 49.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.6
mean_100ep_reward_a2c 49.4
mean_100ep_reward_a2c 49.4
Game has started.
Game has started.
mean_100ep_reward_a2c 49.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.3
mean_100ep_reward_a2c 49.3
mean_100ep_reward_a2c 49.4
Game has started.
Game has started.
mean_100ep_reward_a2c 49.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.3
mean_100ep_reward_a2c 49.2
mean_100ep_reward_a2c 49.1
Game has started.
Game has started.
mean_100ep_reward_a2c 49.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.0
mean_100ep_reward_a2c 49.0
mean_100ep_reward_a2c 48.9
Game has started.
Game has started.
mean_100ep_reward_a2c 48.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.6
mean_100ep_reward_a2c 48.4
mean_100ep_reward_a2c 48.2
Game has started.
Game has started.
mean_100ep_reward_a2c 48.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.0
mean_100ep_reward_a2c 47.9
mean_100ep_reward_a2c 47.9
Game has started.
Game has started.
mean_100ep_reward_a2c 47.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.6
mean_100ep_reward_a2c 47.6
mean_100ep_reward_a2c 47.5
Game has started.
Game has started.
mean_100ep_reward_a2c 47.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.0
mean_100ep_reward_a2c 46.9
mean_100ep_reward_a2c 46.7
Game has started.
Game has started.
mean_100ep_reward_a2c 46.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.4
mean_100ep_reward_a2c 46.4
mean_100ep_reward_a2c 46.4
Game has started.
Game has started.
mean_100ep_reward_a2c 46.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.3
mean_100ep_reward_a2c 46.2
mean_100ep_reward_a2c 46.3
Game has started.
Game has started.
mean_100ep_reward_a2c 46.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.5
mean_100ep_reward_a2c 46.9
mean_100ep_reward_a2c 47.1
Game has started.
Game has started.
mean_100ep_reward_a2c 47.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.6
mean_100ep_reward_a2c 47.8
mean_100ep_reward_a2c 47.7
Game has started.
Game has started.
mean_100ep_reward_a2c 47.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.7
mean_100ep_reward_a2c 47.7
mean_100ep_reward_a2c 47.9
Game has started.
Game has started.
mean_100ep_reward_a2c 48.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.0
mean_100ep_reward_a2c 48.0
mean_100ep_reward_a2c 47.8
Game has started.
Game has started.
mean_100ep_reward_a2c 47.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.1
mean_100ep_reward_a2c 48.3
mean_100ep_reward_a2c 48.5
Game has started.
Game has started.
mean_100ep_reward_a2c 48.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.3
mean_100ep_reward_a2c 48.4
mean_100ep_reward_a2c 48.4
Game has started.
Game has started.
mean_100ep_reward_a2c 48.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.1
mean_100ep_reward_a2c 49.5
mean_100ep_reward_a2c 49.9
Game has started.
Game has started.
mean_100ep_reward_a2c 50.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.7
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
mean_100ep_reward_a2c 53.5
mean_100ep_reward_a2c 53.5
Game has started.
Game has started.
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 54.0
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
mean_100ep_reward_a2c 54.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 54.8
Game has started.
Game has started.
mean_100ep_reward_a2c 54.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 53.8
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.6
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.2
mean_100ep_reward_a2c 53.2
Game has started.
Game has started.
mean_100ep_reward_a2c 53.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
mean_100ep_reward_a2c 53.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.2
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.0
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
mean_100ep_reward_a2c 52.9
Max game loops reached. Current: 263176 Max: 262144
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Performing a full restart of the game.
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.4
mean_100ep_reward_a2c 53.5
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.6
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.0
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.5
Game has started.
Game has started.
mean_100ep_reward_a2c 53.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.2
mean_100ep_reward_a2c 53.2
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.2
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.3
mean_100ep_reward_a2c 50.3
Game has started.
Game has started.
mean_100ep_reward_a2c 50.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.5
mean_100ep_reward_a2c 50.3
mean_100ep_reward_a2c 50.2
Game has started.
Game has started.
mean_100ep_reward_a2c 50.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.4
mean_100ep_reward_a2c 50.6
mean_100ep_reward_a2c 50.7
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
mean_100ep_reward_a2c 50.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.3
mean_100ep_reward_a2c 49.9
mean_100ep_reward_a2c 49.6
Game has started.
Game has started.
mean_100ep_reward_a2c 49.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.9
mean_100ep_reward_a2c 48.7
mean_100ep_reward_a2c 48.1
Game has started.
Game has started.
mean_100ep_reward_a2c 47.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.5
mean_100ep_reward_a2c 47.4
mean_100ep_reward_a2c 47.1
Game has started.
Game has started.
mean_100ep_reward_a2c 47.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.8
mean_100ep_reward_a2c 46.6
mean_100ep_reward_a2c 46.2
Game has started.
Game has started.
mean_100ep_reward_a2c 46.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.0
mean_100ep_reward_a2c 46.0
mean_100ep_reward_a2c 45.9
Game has started.
Game has started.
mean_100ep_reward_a2c 45.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.8
mean_100ep_reward_a2c 45.7
mean_100ep_reward_a2c 45.8
Game has started.
Game has started.
mean_100ep_reward_a2c 45.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.6
mean_100ep_reward_a2c 45.5
mean_100ep_reward_a2c 45.5
Game has started.
Game has started.
mean_100ep_reward_a2c 45.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.4
mean_100ep_reward_a2c 45.5
mean_100ep_reward_a2c 45.4
Game has started.
Game has started.
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.1
mean_100ep_reward_a2c 45.2
mean_100ep_reward_a2c 45.2
Game has started.
Game has started.
mean_100ep_reward_a2c 44.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.4
mean_100ep_reward_a2c 44.1
mean_100ep_reward_a2c 43.9
Game has started.
Game has started.
mean_100ep_reward_a2c 43.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 43.5
mean_100ep_reward_a2c 43.3
mean_100ep_reward_a2c 43.2
Game has started.
Game has started.
mean_100ep_reward_a2c 43.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 43.2
mean_100ep_reward_a2c 43.5
mean_100ep_reward_a2c 43.8
Game has started.
Game has started.
mean_100ep_reward_a2c 43.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 43.2
mean_100ep_reward_a2c 42.9
mean_100ep_reward_a2c 42.6
Game has started.
Game has started.
mean_100ep_reward_a2c 42.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 42.6
mean_100ep_reward_a2c 42.6
mean_100ep_reward_a2c 42.6
Game has started.
Game has started.
mean_100ep_reward_a2c 42.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 42.9
mean_100ep_reward_a2c 42.8
mean_100ep_reward_a2c 42.6
Game has started.
Game has started.
mean_100ep_reward_a2c 42.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 42.4
mean_100ep_reward_a2c 42.1
mean_100ep_reward_a2c 41.9
Game has started.
Game has started.
mean_100ep_reward_a2c 41.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 41.8
mean_100ep_reward_a2c 41.8
mean_100ep_reward_a2c 41.7
Game has started.
Game has started.
mean_100ep_reward_a2c 41.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 41.9
mean_100ep_reward_a2c 42.0
mean_100ep_reward_a2c 42.1
Game has started.
Game has started.
mean_100ep_reward_a2c 42.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 42.4
mean_100ep_reward_a2c 42.7
mean_100ep_reward_a2c 42.9
Game has started.
Game has started.
mean_100ep_reward_a2c 43.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 43.3
mean_100ep_reward_a2c 43.8
mean_100ep_reward_a2c 44.1
Game has started.
Game has started.
mean_100ep_reward_a2c 44.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.0
mean_100ep_reward_a2c 44.1
mean_100ep_reward_a2c 44.2
Game has started.
Game has started.
mean_100ep_reward_a2c 44.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.7
mean_100ep_reward_a2c 44.7
mean_100ep_reward_a2c 44.8
Game has started.
Game has started.
mean_100ep_reward_a2c 44.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.0
mean_100ep_reward_a2c 45.0
mean_100ep_reward_a2c 44.8
Game has started.
Game has started.
mean_100ep_reward_a2c 45.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.1
mean_100ep_reward_a2c 45.1
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.4
mean_100ep_reward_a2c 45.6
mean_100ep_reward_a2c 45.9
Game has started.
Game has started.
mean_100ep_reward_a2c 46.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.4
mean_100ep_reward_a2c 46.6
mean_100ep_reward_a2c 46.8
Game has started.
Game has started.
mean_100ep_reward_a2c 47.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.3
mean_100ep_reward_a2c 47.6
mean_100ep_reward_a2c 47.7
Game has started.
Game has started.
mean_100ep_reward_a2c 48.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.1
mean_100ep_reward_a2c 48.4
mean_100ep_reward_a2c 48.4
Game has started.
Game has started.
mean_100ep_reward_a2c 48.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.6
mean_100ep_reward_a2c 48.9
mean_100ep_reward_a2c 48.9
Game has started.
Game has started.
mean_100ep_reward_a2c 49.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.2
mean_100ep_reward_a2c 49.4
mean_100ep_reward_a2c 49.6
Game has started.
Game has started.
mean_100ep_reward_a2c 50.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.0
mean_100ep_reward_a2c 50.0
mean_100ep_reward_a2c 50.0
Game has started.
Game has started.
mean_100ep_reward_a2c 50.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.2
mean_100ep_reward_a2c 50.4
mean_100ep_reward_a2c 50.5
Game has started.
Game has started.
mean_100ep_reward_a2c 50.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.3
mean_100ep_reward_a2c 50.3
mean_100ep_reward_a2c 50.1
Game has started.
Game has started.
mean_100ep_reward_a2c 50.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.2
mean_100ep_reward_a2c 50.3
mean_100ep_reward_a2c 50.3
Game has started.
Game has started.
mean_100ep_reward_a2c 50.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
mean_100ep_reward_a2c 51.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.6
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 52.0
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.2
mean_100ep_reward_a2c 53.2
mean_100ep_reward_a2c 53.2
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.5
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
mean_100ep_reward_a2c 53.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 53.8
Game has started.
Game has started.
mean_100ep_reward_a2c 53.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.8
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.7
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.7
mean_100ep_reward_a2c 56.0
mean_100ep_reward_a2c 56.2
Game has started.
Game has started.
mean_100ep_reward_a2c 56.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.5
mean_100ep_reward_a2c 56.6
mean_100ep_reward_a2c 56.5
Game has started.
Game has started.
mean_100ep_reward_a2c 56.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.2
mean_100ep_reward_a2c 56.0
mean_100ep_reward_a2c 56.0
Game has started.
Game has started.
mean_100ep_reward_a2c 56.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.2
mean_100ep_reward_a2c 56.3
mean_100ep_reward_a2c 56.5
Game has started.
Game has started.
mean_100ep_reward_a2c 56.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.9
mean_100ep_reward_a2c 57.0
mean_100ep_reward_a2c 56.9
Game has started.
Game has started.
mean_100ep_reward_a2c 57.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.1
mean_100ep_reward_a2c 57.0
mean_100ep_reward_a2c 57.2
Game has started.
Game has started.
mean_100ep_reward_a2c 57.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.4
mean_100ep_reward_a2c 57.4
mean_100ep_reward_a2c 57.5
Game has started.
Game has started.
mean_100ep_reward_a2c 57.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.3
mean_100ep_reward_a2c 57.3
mean_100ep_reward_a2c 57.3
Game has started.
Game has started.
mean_100ep_reward_a2c 57.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.1
Game has started.
Game has started.
mean_100ep_reward_a2c 56.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.9
mean_100ep_reward_a2c 56.7
mean_100ep_reward_a2c 56.7
Game has started.
Game has started.
mean_100ep_reward_a2c 56.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.9
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.2
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.2
mean_100ep_reward_a2c 57.3
mean_100ep_reward_a2c 57.3
Game has started.
Game has started.
mean_100ep_reward_a2c 57.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.4
mean_100ep_reward_a2c 57.4
mean_100ep_reward_a2c 57.4
Game has started.
Game has started.
mean_100ep_reward_a2c 57.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.6
mean_100ep_reward_a2c 57.9
mean_100ep_reward_a2c 58.1
Game has started.
Game has started.
mean_100ep_reward_a2c 58.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.4
mean_100ep_reward_a2c 58.3
mean_100ep_reward_a2c 58.4
Game has started.
Game has started.
mean_100ep_reward_a2c 58.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.7
mean_100ep_reward_a2c 58.9
mean_100ep_reward_a2c 59.2
Game has started.
Game has started.
mean_100ep_reward_a2c 59.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.5
mean_100ep_reward_a2c 59.4
mean_100ep_reward_a2c 59.7
Game has started.
Game has started.
mean_100ep_reward_a2c 59.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.5
mean_100ep_reward_a2c 59.5
mean_100ep_reward_a2c 59.4
Game has started.
Game has started.
mean_100ep_reward_a2c 59.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 59.4
mean_100ep_reward_a2c 59.2
mean_100ep_reward_a2c 58.8
Game has started.
Game has started.
mean_100ep_reward_a2c 58.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.6
mean_100ep_reward_a2c 58.4
mean_100ep_reward_a2c 58.4
Game has started.
Game has started.
mean_100ep_reward_a2c 58.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 58.2
mean_100ep_reward_a2c 58.1
mean_100ep_reward_a2c 57.9
Game has started.
Game has started.
mean_100ep_reward_a2c 57.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.5
mean_100ep_reward_a2c 57.3
mean_100ep_reward_a2c 57.1
Game has started.
Game has started.
mean_100ep_reward_a2c 57.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 57.1
mean_100ep_reward_a2c 57.1
mean_100ep_reward_a2c 56.8
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
mean_100ep_reward_a2c 56.6
mean_100ep_reward_a2c 56.4
Game has started.
Game has started.
mean_100ep_reward_a2c 56.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.2
mean_100ep_reward_a2c 56.0
mean_100ep_reward_a2c 56.0
Game has started.
Game has started.
mean_100ep_reward_a2c 55.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
mean_100ep_reward_a2c 54.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
mean_100ep_reward_a2c 53.2
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.6
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.8
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.2
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.9
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.4
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.5
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
mean_100ep_reward_a2c 53.4
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.7
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
mean_100ep_reward_a2c 53.5
mean_100ep_reward_a2c 53.6
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.6
Game has started.
Game has started.
mean_100ep_reward_a2c 54.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.7
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
mean_100ep_reward_a2c 54.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.6
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
mean_100ep_reward_a2c 54.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.9
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.4
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
mean_100ep_reward_a2c 51.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.7
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.7
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.6
mean_100ep_reward_a2c 50.4
Game has started.
Game has started.
mean_100ep_reward_a2c 50.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.5
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.5
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
mean_100ep_reward_a2c 51.6
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Game has started.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.2
Game has started.
Game has started.
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.7
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.1
mean_100ep_reward_a2c 53.5
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.6
mean_100ep_reward_a2c 54.7
mean_100ep_reward_a2c 54.8
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
mean_100ep_reward_a2c 54.9
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.6
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.8
mean_100ep_reward_a2c 55.7
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.6
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
mean_100ep_reward_a2c 54.5
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.8
mean_100ep_reward_a2c 53.6
mean_100ep_reward_a2c 53.2
Game has started.
Game has started.
mean_100ep_reward_a2c 53.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.8
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.4
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.7
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
mean_100ep_reward_a2c 52.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.7
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.6
Game has started.
Game has started.
mean_100ep_reward_a2c 50.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.5
mean_100ep_reward_a2c 50.3
mean_100ep_reward_a2c 50.2
Game has started.
Game has started.
mean_100ep_reward_a2c 50.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.2
mean_100ep_reward_a2c 50.2
mean_100ep_reward_a2c 50.4
Game has started.
Game has started.
mean_100ep_reward_a2c 50.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.6
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.7
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.7
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.8
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.7
mean_100ep_reward_a2c 52.0
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.5
mean_100ep_reward_a2c 52.9
Game has started.
Game has started.
mean_100ep_reward_a2c 53.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.7
mean_100ep_reward_a2c 54.0
mean_100ep_reward_a2c 54.2
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.5
mean_100ep_reward_a2c 54.7
mean_100ep_reward_a2c 54.9
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.1
mean_100ep_reward_a2c 55.3
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.7
mean_100ep_reward_a2c 55.6
mean_100ep_reward_a2c 55.8
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.8
mean_100ep_reward_a2c 55.7
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
mean_100ep_reward_a2c 55.5
mean_100ep_reward_a2c 55.3
Game has started.
Game has started.
mean_100ep_reward_a2c 55.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.7
mean_100ep_reward_a2c 55.8
mean_100ep_reward_a2c 55.8
Game has started.
Game has started.
mean_100ep_reward_a2c 55.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.0
mean_100ep_reward_a2c 56.0
mean_100ep_reward_a2c 56.2
Game has started.
Game has started.
mean_100ep_reward_a2c 56.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.4
mean_100ep_reward_a2c 56.5
mean_100ep_reward_a2c 56.8
Game has started.
Game has started.
mean_100ep_reward_a2c 56.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 56.3
mean_100ep_reward_a2c 56.1
mean_100ep_reward_a2c 55.8
Game has started.
Game has started.
mean_100ep_reward_a2c 55.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.6
mean_100ep_reward_a2c 55.4
mean_100ep_reward_a2c 55.4
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 55.2
mean_100ep_reward_a2c 54.8
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
mean_100ep_reward_a2c 55.0
mean_100ep_reward_a2c 55.0
Game has started.
Game has started.
mean_100ep_reward_a2c 54.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
mean_100ep_reward_a2c 54.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
mean_100ep_reward_a2c 54.4
mean_100ep_reward_a2c 54.5
Game has started.
Game has started.
mean_100ep_reward_a2c 54.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 54.2
mean_100ep_reward_a2c 54.1
mean_100ep_reward_a2c 54.0
Game has started.
Game has started.
mean_100ep_reward_a2c 53.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 53.4
mean_100ep_reward_a2c 53.0
mean_100ep_reward_a2c 52.7
Game has started.
Game has started.
mean_100ep_reward_a2c 52.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.0
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.8
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.7
mean_100ep_reward_a2c 50.8
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.1
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 50.9
Game has started.
Game has started.
mean_100ep_reward_a2c 50.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.4
mean_100ep_reward_a2c 50.6
mean_100ep_reward_a2c 50.6
Game has started.
Game has started.
mean_100ep_reward_a2c 50.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.9
mean_100ep_reward_a2c 49.7
mean_100ep_reward_a2c 49.4
Game has started.
Game has started.
mean_100ep_reward_a2c 49.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.9
mean_100ep_reward_a2c 48.5
mean_100ep_reward_a2c 48.2
Game has started.
Game has started.
mean_100ep_reward_a2c 48.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.8
mean_100ep_reward_a2c 47.4
mean_100ep_reward_a2c 47.3
Game has started.
Game has started.
mean_100ep_reward_a2c 47.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.1
mean_100ep_reward_a2c 47.1
mean_100ep_reward_a2c 47.2
Game has started.
Game has started.
mean_100ep_reward_a2c 47.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.9
mean_100ep_reward_a2c 46.6
mean_100ep_reward_a2c 46.4
Game has started.
Game has started.
mean_100ep_reward_a2c 46.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.2
mean_100ep_reward_a2c 46.2
mean_100ep_reward_a2c 46.3
Game has started.
Game has started.
mean_100ep_reward_a2c 46.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.6
mean_100ep_reward_a2c 46.8
mean_100ep_reward_a2c 46.9
Game has started.
Game has started.
mean_100ep_reward_a2c 47.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.1
mean_100ep_reward_a2c 47.4
mean_100ep_reward_a2c 47.7
Game has started.
Game has started.
mean_100ep_reward_a2c 48.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.9
mean_100ep_reward_a2c 47.6
mean_100ep_reward_a2c 47.7
Game has started.
Game has started.
mean_100ep_reward_a2c 47.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.7
mean_100ep_reward_a2c 47.7
mean_100ep_reward_a2c 47.7
Game has started.
Game has started.
mean_100ep_reward_a2c 47.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.9
mean_100ep_reward_a2c 47.8
mean_100ep_reward_a2c 47.8
Game has started.
Game has started.
mean_100ep_reward_a2c 47.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.9
mean_100ep_reward_a2c 48.0
mean_100ep_reward_a2c 48.1
Game has started.
Game has started.
mean_100ep_reward_a2c 48.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.6
mean_100ep_reward_a2c 48.7
mean_100ep_reward_a2c 48.8
Game has started.
Game has started.
mean_100ep_reward_a2c 48.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.8
mean_100ep_reward_a2c 48.9
mean_100ep_reward_a2c 49.0
Game has started.
Game has started.
mean_100ep_reward_a2c 49.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.9
mean_100ep_reward_a2c 49.1
mean_100ep_reward_a2c 49.1
Game has started.
Game has started.
mean_100ep_reward_a2c 49.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.5
mean_100ep_reward_a2c 49.9
mean_100ep_reward_a2c 50.1
Game has started.
Game has started.
mean_100ep_reward_a2c 50.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.4
mean_100ep_reward_a2c 50.5
mean_100ep_reward_a2c 50.5
Game has started.
Game has started.
mean_100ep_reward_a2c 50.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
mean_100ep_reward_a2c 51.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.2
mean_100ep_reward_a2c 52.3
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 52.1
mean_100ep_reward_a2c 51.9
mean_100ep_reward_a2c 51.9
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.3
mean_100ep_reward_a2c 50.8
mean_100ep_reward_a2c 50.6
Game has started.
Game has started.
mean_100ep_reward_a2c 50.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.9
mean_100ep_reward_a2c 49.6
mean_100ep_reward_a2c 49.5
Game has started.
Game has started.
mean_100ep_reward_a2c 49.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.3
mean_100ep_reward_a2c 49.5
mean_100ep_reward_a2c 49.6
Game has started.
Game has started.
mean_100ep_reward_a2c 49.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 49.7
mean_100ep_reward_a2c 50.0
mean_100ep_reward_a2c 49.8
Game has started.
Game has started.
mean_100ep_reward_a2c 49.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.1
mean_100ep_reward_a2c 50.2
mean_100ep_reward_a2c 50.5
Game has started.
Game has started.
mean_100ep_reward_a2c 50.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.9
mean_100ep_reward_a2c 51.0
mean_100ep_reward_a2c 51.2
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.5
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.7
mean_100ep_reward_a2c 51.6
Game has started.
Game has started.
mean_100ep_reward_a2c 51.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.6
mean_100ep_reward_a2c 51.8
mean_100ep_reward_a2c 51.7
Game has started.
Game has started.
mean_100ep_reward_a2c 51.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.5
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 51.4
mean_100ep_reward_a2c 51.2
mean_100ep_reward_a2c 51.0
Game has started.
Game has started.
mean_100ep_reward_a2c 50.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 50.0
mean_100ep_reward_a2c 49.7
mean_100ep_reward_a2c 49.4
Game has started.
Game has started.
mean_100ep_reward_a2c 49.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.7
mean_100ep_reward_a2c 48.3
mean_100ep_reward_a2c 47.9
Game has started.
Game has started.
mean_100ep_reward_a2c 47.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.4
mean_100ep_reward_a2c 47.1
mean_100ep_reward_a2c 46.9
Game has started.
Game has started.
mean_100ep_reward_a2c 46.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.5
mean_100ep_reward_a2c 46.2
mean_100ep_reward_a2c 46.1
Game has started.
Game has started.
mean_100ep_reward_a2c 46.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.1
mean_100ep_reward_a2c 46.0
mean_100ep_reward_a2c 45.9
Game has started.
Game has started.
mean_100ep_reward_a2c 45.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.8
mean_100ep_reward_a2c 45.8
mean_100ep_reward_a2c 45.8
Game has started.
Game has started.
mean_100ep_reward_a2c 45.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.0
mean_100ep_reward_a2c 46.2
mean_100ep_reward_a2c 46.0
Game has started.
Game has started.
mean_100ep_reward_a2c 45.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.6
mean_100ep_reward_a2c 45.6
mean_100ep_reward_a2c 45.5
Game has started.
Game has started.
mean_100ep_reward_a2c 45.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.4
mean_100ep_reward_a2c 45.4
mean_100ep_reward_a2c 45.4
Game has started.
Game has started.
mean_100ep_reward_a2c 45.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.2
mean_100ep_reward_a2c 44.9
mean_100ep_reward_a2c 44.9
Game has started.
Game has started.
mean_100ep_reward_a2c 44.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.7
mean_100ep_reward_a2c 44.9
mean_100ep_reward_a2c 45.1
Game has started.
Game has started.
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.4
mean_100ep_reward_a2c 45.6
mean_100ep_reward_a2c 45.8
Game has started.
Game has started.
mean_100ep_reward_a2c 46.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.9
mean_100ep_reward_a2c 45.8
mean_100ep_reward_a2c 45.7
Game has started.
Game has started.
mean_100ep_reward_a2c 45.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.5
mean_100ep_reward_a2c 45.3
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
mean_100ep_reward_a2c 45.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.6
mean_100ep_reward_a2c 46.0
mean_100ep_reward_a2c 46.5
Game has started.
Game has started.
mean_100ep_reward_a2c 46.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.7
mean_100ep_reward_a2c 47.0
mean_100ep_reward_a2c 47.3
Game has started.
Game has started.
mean_100ep_reward_a2c 47.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.6
mean_100ep_reward_a2c 47.5
mean_100ep_reward_a2c 47.4
Game has started.
Game has started.
mean_100ep_reward_a2c 47.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.4
mean_100ep_reward_a2c 47.4
mean_100ep_reward_a2c 47.5
Game has started.
Game has started.
mean_100ep_reward_a2c 47.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.1
mean_100ep_reward_a2c 48.1
mean_100ep_reward_a2c 48.3
Game has started.
Game has started.
mean_100ep_reward_a2c 48.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.5
mean_100ep_reward_a2c 48.6
mean_100ep_reward_a2c 48.6
Game has started.
Game has started.
mean_100ep_reward_a2c 48.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.6
mean_100ep_reward_a2c 48.6
mean_100ep_reward_a2c 48.7
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
mean_100ep_reward_a2c 48.5
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Max game loops reached. Current: 263176 Max: 262144
Performing a full restart of the game.
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.5
mean_100ep_reward_a2c 48.3
mean_100ep_reward_a2c 48.2
Game has started.
Game has started.
mean_100ep_reward_a2c 48.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.1
mean_100ep_reward_a2c 48.2
mean_100ep_reward_a2c 48.1
Game has started.
Game has started.
mean_100ep_reward_a2c 48.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.7
mean_100ep_reward_a2c 47.5
mean_100ep_reward_a2c 47.3
Game has started.
Game has started.
mean_100ep_reward_a2c 47.0
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.9
mean_100ep_reward_a2c 46.8
mean_100ep_reward_a2c 46.6
Game has started.
Game has started.
mean_100ep_reward_a2c 46.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.7
mean_100ep_reward_a2c 46.9
mean_100ep_reward_a2c 47.1
Game has started.
Game has started.
mean_100ep_reward_a2c 47.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.8
mean_100ep_reward_a2c 48.2
mean_100ep_reward_a2c 48.7
Game has started.
Game has started.
mean_100ep_reward_a2c 48.7
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.5
mean_100ep_reward_a2c 48.3
mean_100ep_reward_a2c 48.1
init group list
init group list
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.5
mean_100ep_reward_a2c 48.4
mean_100ep_reward_a2c 48.6
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.7
mean_100ep_reward_a2c 48.8
mean_100ep_reward_a2c 48.8
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.8
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.6
mean_100ep_reward_a2c 48.6
mean_100ep_reward_a2c 48.6
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.4
mean_100ep_reward_a2c 48.2
mean_100ep_reward_a2c 48.3
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.4
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.4
mean_100ep_reward_a2c 48.4
mean_100ep_reward_a2c 48.3
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.2
mean_100ep_reward_a2c 48.2
mean_100ep_reward_a2c 48.0
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.9
mean_100ep_reward_a2c 48.0
mean_100ep_reward_a2c 48.2
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.1
mean_100ep_reward_a2c 48.3
mean_100ep_reward_a2c 48.2
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 48.1
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.9
mean_100ep_reward_a2c 47.8
mean_100ep_reward_a2c 47.6
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.4
mean_100ep_reward_a2c 47.6
mean_100ep_reward_a2c 47.8
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.5
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.5
mean_100ep_reward_a2c 47.6
mean_100ep_reward_a2c 47.4
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.6
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 47.8
mean_100ep_reward_a2c 47.6
mean_100ep_reward_a2c 47.3
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.8
mean_100ep_reward_a2c 46.8
mean_100ep_reward_a2c 46.4
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 46.2
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.9
mean_100ep_reward_a2c 45.6
mean_100ep_reward_a2c 45.4
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.4
mean_100ep_reward_a2c 45.2
mean_100ep_reward_a2c 45.2
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.2
mean_100ep_reward_a2c 45.3
mean_100ep_reward_a2c 45.2
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.3
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.1
mean_100ep_reward_a2c 45.2
mean_100ep_reward_a2c 45.1
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 44.9
Game has started.
Game has started.
Game has started.
Game has started.
Game has started.
mean_100ep_reward_a2c 45.0
mean_100ep_reward_a2c 44.6
mean_100ep_reward_a2c 44.5
</div>

## 2. PySC2使用体验（对比Win下C++项目）
在Linux下使用了PySC2几天后的主要感受如下：
### 游戏的UI界面
DeepMind开发的UI主要分为主要两部分：左边的***minimap***和右边的***screen***：
***minimap***是整个游戏界面的视角，而***screen***是从不同维度观测的特征视角，如地图全貌，探测过的地图，玩家类型，单位类型等。这些Feature map类似卷积网络中的Feature Layer，事实上我也认为QDN的实现就是基于此

![](https://image.ibb.co/iJ0mWx/pysc2_ui.png) 

此外，这个UI是2D角度观测的，不同与Win平台和Mac平台，Linux下的游戏没有RGB渲染，也没有3D视角，根据PySC2的[paper](https://deepmind.com/documents/110/sc2le.pdf)所述，这样会导致PySC2的Bot与人类的游戏视角有区别，由于3D视角较高，会导致Bot看到前面多一些而背面少一些，因此人类玩家的一些操作可能不能被完全复制
个人感受较深的是这个UI下<span style="color:red">很难进行玩家的操作</span>，比如点击单位，执行命令等，相比Win下的CommandCenter项目显得不直观，并且难以操作
### 启动命令与依赖问题
1.启动命令 PySC2是在terminal里执行启动命令，如
``` sh
python3 -m pysc2.bin.agent --map Simple64 --agent simple_agent.SimpleAgent --agent_race T
```
比较简洁明了，而对比之前在VS2017中执行C++的启动命令则显得臃肿

![](https://image.ibb.co/fZtJdc/c_command_line.png)

2.三方依赖 python项目的依赖问题在pycharm里点击自动安装依赖包即可，或者手动在设置里添加，因为用pip统一管理，我认为相对VS中C++项目的依赖问题更容易解决，也清晰明了。（之前由于.NET依赖未添加导致CommandCenter总是编译失败，但是具体缺失的依赖项难以追溯，耽误了我很长时间，而PyCharm里集中管理就容易追溯到缺失的三方包）

![](https://image.ibb.co/kSApJc/Py_Charm_Package.png)


### 代码对比（Python vs C++）
个人认为Python相比C++最大的两点优势如下：

- 可读性很强，逻辑清晰
- 即时编译，便于Debug

以SimpleAgent的python代码为例
``` python
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time

# Functions
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_BARRACKS = 21

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]
_SUPPLY_USED = 3
_SUPPLY_MAX = 4


class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None
    supply_depot_built = False
    barracks_built = False
    scv_selected = False
    barracks_selected = False
    barracks_rallied = False
    army_selected = False
    army_rallied = False

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        # time.sleep(0.2)

        if self.base_top_left is None:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31

        if not self.supply_depot_built:
            if not self.scv_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                target = [unit_x[0], unit_y[0]]

                self.scv_selected = True

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

                self.supply_depot_built = True

                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])

        elif not self.barracks_built:
            if _BUILD_BARRACKS in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                self.barracks_built = True

                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        elif not self.barracks_rallied:
            if not self.barracks_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]

                    self.barracks_selected = True

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            else:
                self.barracks_rallied = True

                if self.base_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])

                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])

        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_MARINE in \
                obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False

                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                self.army_rallied = True
                self.army_selected = False

                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])

                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])

        return actions.FunctionCall(_NOOP, [])
```
在class上方清晰的给出了不同的全局变量定义，Function，Feature，Unit ID和Parameter，并且主函数step中的逻辑也十分明朗，除了unit_y, unit_x的获取是与直观逻辑相悖外，操作都是以人类逻辑的原子操作为标准。
而C++的项目中step函数显得没那么清晰：
``` c++
void CCBot::OnStep()
{
    setUnits();
    m_map.onFrame();
    m_unitInfo.onFrame();
    m_bases.onFrame();
    m_workers.onFrame();
    m_strategy.onFrame();

    m_gameCommander.onFrame();

#ifdef SC2API
    Debug()->SendDebug();
#endif
}
```
这里仅以CCBot.cpp为例，因为类之间的调用层级较深，没有PySC2里清晰，并且有些变量会在.h头文件里定义，有些在s2c-client里定义等
并且由于typedef的存在，c++项目里开发不看得很清晰就难以开发，相对而言，python的duck传参就易于开发

另外pysc2比较容易修改code，并且traceback我认为也比vs里的traceback好用，因此调试过程相对而言更轻松一些

--------
但是python相比c++也有一些<span style="color:red">不足</span>：主要是代码的缩进上，由于函数体或者循环块完全由缩进决定，导致有时候由于缩进导致的bug难以发现，如果用花括号{}就难以出现这个问题
至于执行效率上，听说python近两年通过编译c文件大大提高了执行效率，具体没有和c++对比，但是我想开发RL的话，逻辑性更强的语言应该会好一些