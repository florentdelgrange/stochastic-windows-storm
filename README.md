# StormWind 
## Installation
### Requirements
Make sure you have set up all dependencies as required [here](http://www.stormchecker.org/documentation/installation/requirements.html).
Depending of your OS version, you may have some issues with the installation of Boost or Carl.
You can install them manually as described [here](http://www.stormchecker.org/documentation/installation/manual-configuration.html).
### Storm sources
Get the original [Storm](http://www.stormchecker.org/index.html) source files from github as follows.
```
git clone https://github.com/moves-rwth/storm.git
```
### Build the storm *window* module
```
cd storm/src
git init .
git remote add stochastic-windows https://github.com/theGreatGiorgio/stochastic-windows-storm.git
git fetch --all
git reset --hard stochastic-windows/master
cp -R test/stochastic-windows/resources/* ../resources/examples/testfiles/
mkdir ../build
cd ../build
cmake ..
make
```

## Run
The executable is located in the binary directory `build`. You can display the options of the window module by typping
```
./storm-stochastic-windows --help sw
```
These options are the following:
```
--[sw:]objective <window variant> <long run objective> Sets which window objective to consider. <window variant> (in {dfw, fw, bw}): The window variant to consider. <long run objective> (in {mp, par}): The long run objective to be strengthened with the window mecanism.
--[sw:]rew <reward model name> ....................... The reward model to consider for the window objective <reward model name>: the name of the reward model to consider
--[sw:]windowsize <value> ............................ Sets the maximal window size. <value> (in [1, +inf)): The window size.
--[sw:]exportdotscheduler <filename> ................. If given, a scheduler maximizing the probability of the window objective will be written to the specified file in the dot format. <filename>: The name of the file to which the scheduler is to be written.
--[sw:]classificationmethod <name> ................... If given, sets the maximal end component classification method. <name> (in {default, bwgame, dfwgame, unfolding}; default: default): The name of the method to use during the maximal end component classification.
--[sw:]schedulerslabels .............................. Adds labels to the synthesized scheduler.
```

Note that you can also display the I/O options by typping
```
./storm-stochastic-windows --help io
```

### Synthesis
The strategy/scheduler synthesis is only available for the policy iteration solver (you can set it with the option `--minmax:method pi`). You can retrieve the strategy in a text file via the option `--io:exportscheduler` or via in a dotfile via the option `exportdotscheduler`.

### Example

```
./storm-stochastic-windows --minmax:method pi --prism "mdp.nm" --objective bw mp --rew "weights" --windowsize 3 --io:exportscheduler "scheduler.txt" --io:exportdot "mdp.dot" --exportdotscheduler "scheduler.dot" --schedulerslabels
```
This will display the maximal probability of satisfying the bounded window mean payoff objective of maximal window size 3 in the MDP encoded in the prism file `mdp.nm`, where the weights are encoded in the reward model of name `weights`. This will also write the input model to the dot file `mdp.dot` and write the linked strategy to the file `/scheduler.txt` and to the dot file `scheduler.dot` by displaying the labels of its memory states in this dot file (via the option `--schedulerslabels`). One can then visualize the strategy in pdf format with graphviz as follows:
```
dot -Tpdf scheduler.dot -o scheduler.pdf
```
