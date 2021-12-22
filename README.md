# pawn

git clone git@github.com:wutas/pawn.git

scp -r models 

sudo docker build -t wutas/pawnshop .

sudo docker run -d -p 8001:8501 wutas/pawnshop
