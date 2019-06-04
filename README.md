# StormWind 
## Installation
### Requirements
Make sure that you have set up all dependencies as required [here](http://www.stormchecker.org/documentation/installation/requirements.html).
Regarding to your OS version, you may have some issues with the installation of Boost or Carl.
You can install them manually as described [here](http://www.stormchecker.org/documentation/installation/manual-configuration.html).
### Storm sources
Get the original [Storm](http://www.stormchecker.org/index.html) source files from github as follows.
```
git clone https://github.com/moves-rwth/storm.git
```
### Build the *Windows* module
```
cd storm/src
git remote add stochastic-windows https://github.com/theGreatGiorgio/stochastic-windows-storm.git
git fetch --all
git reset --hard stochastic-windows/master
mkdir ../build
cd ../build
cmake ..
make
```

## Run
Go the binaries directory and run
```
./storm-stochastic-windows
```
Note that all storm's command line arguments are available.
Note also that the strategy/scheduler synthesis is only available for the policy iteration solver.
To set it, run
```
./storm-stochastic-windows --minmax:method pi
```
