# Beer Game demo
Beer Game implemented as an OpenAI Gym environment.


![roles.png](docs%2Froles.png)


![order_shipment.png](docs%2Forder_shipment.png)


Installation:

1. Create a new conda environment to keep things clean
```
conda create python=3.7.5 --name beer-game-env
source activate beer-game-env
```

2. Clone the environment repository
```
git clone https://github.com/YanSong97/BeerGame_demo.git 
```

3. Point to root repository and install the package
```
cd BeerGame_demo
pip install -e .
```

To use:
```
python new_env.py
```

tested with gym version `gym==0.26.2`