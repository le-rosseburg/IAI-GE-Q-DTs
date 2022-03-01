# IAI-GE-Q-DTs
Reimplementation of the paper ["Evolutionary learning of interpretable decision trees"](https://arxiv.org/pdf/2012.07723v3.pdf) by Leonardo Lucio Custode, Giovanni Iacca.

---

## Installation guide
- pip install git+https://github.com/maotto/deap@master

for LunarLander-v2:
- conda install swig
- pip install gym[Box2D]

---

## Commands
### CartPole
##### Orthogonal
python test_evolution.py --grammar orthogonal --environment_name CartPole-v1 --seed 42 --n_actions 2 --learning_rate 0.001 --df 0.05 --input_space 4 --episodes 10 --population_size 200 --generations 100 --cxp 0 --mp 1 --low -1 --up 1 --genotype_len 1024 --types #-48,48,5,10;-50,50,5,10;-418,418,5,1000;-836,836,5,1000 --mutation "function-tools.mutUniformInt#low-0#up-40000#indpb-0.1"

##### Oblique
python test_evolution.py --grammar oblique --environment_name CartPole-v1 --seed 42 --n_actions 4 --learning_rate 0.001 --df 0.05 --input_space 4 --episodes 10 --population_size 200 --generations 50 --cxp 0 --mp 1 --low -1 --up 1 --genotype_len 100 --types #-48,48,5,10;-50,50,5,10;-418,418,5,1000;-836,836,5,1000 --mutation "function-tools.mutUniformInt#low-0#up-4000#indpb-0.1"

### MountainCar
##### Orthogonal
python test_evolution.py --grammar orthogonal --environment_name MountainCar-v0 --seed 42 --n_actions 3 --learning_rate 0.001 --df 0.05 --input_space 2 --episodes 10 --population_size 200 --generations 1000 --cxp 0 --mp 1 --low -1 --up 1 --genotype_len 1024 --types #-120,60,5,100;-70,70,5,1000 --mutation "function-tools.mutUniformInt#low-0#up-40000#indpb-0.05"

##### Oblique
python test_evolution.py --grammar oblique --environment_name MountainCar-v0 --seed 42 --n_actions 3 --learning_rate 0.001 --df 0.05 --input_space 2 --episodes 10 --population_size 200 --generations 2000 --cxp 0.1 --mp 1 --low -1 --up 1 --eps 0.01 --genotype_len 100 --types #-120,70,5,100;-70,70,5,1000 --mutation "function-tools.mutUniformInt#low-0#up-40000#indpb-0.1"

### LunarLander
##### Oblique
python test_evolution.py --grammar oblique --environment_name LunarLander-v2 --seed 42 --n_actions 4 --learning_rate "auto" --df 0.9 --input_space 8 --episodes 1000 --population_size 100 --generations 100 --cxp 0.1 --mp 1 --low -1 --up 1 --eps 1.0 --genotype_len 100 --randInit False --with_bias False --types #-000,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000 --mutation "function-tools.mutUniformInt#low-0#up-40000#indpb-0.05"

---

## Links
- [Paper](https://arxiv.org/pdf/2012.07723.pdf)
- [Codebase of paper](https://gitlab.com/leocus/ge_q_dts)
- [Wikilink: Grammatical evolution](https://en.wikipedia.org/wiki/Grammatical_evolution)
- [DEAP Library](https://github.com/deap/deap)
- [PonyGE Library](https://github.com/PonyGE/PonyGE2) and [Introduction to PonyGE](https://towardsdatascience.com/introduction-to-ponyge2-for-grammatical-evolution-d51c29f2315a)

## Possible Optimizations: 
- [Particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization):
  -   [Guide](https://www.analyticsvidhya.com/blog/2021/11/implementing-a-particle-swarm-optimization-with-python/)
  -   [PSO from scratch](https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6)

## Project Guidlines: 
- [Reproducibility Checklist](https://studip.uni-hannover.de/sendfile.php?type=0&file_id=a2067dd448cbae4be0ebaabc1809dd1b&file_name=Reproducibility.pdf)
- [PEP8 code documentation](https://www.python.org/dev/peps/pep-0008/)
- [Lecture powerpoint](https://studip.uni-hannover.de/sendfile.php?type=0&file_id=f59cece59252733b699685dd73438268&file_name=RL_lecture_exam_21_22.pdf)
- [DRL that Matters Paper](https://arxiv.org/abs/1709.06560)
- [DRL that Matters Whitboard](https://miro.com/app/board/uXjVOasqpog=/)