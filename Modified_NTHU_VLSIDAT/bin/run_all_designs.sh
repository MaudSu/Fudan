#!/bin/bash
rm ./congestion.csv

cp ./predicted_imgs/congestion_adaptec1.csv ./congestion.csv
./route --input=adaptec1 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_adaptec2.csv ./congestion.csv
./route --input=adaptec2 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_adaptec3.csv ./congestion.csv
./route --input=adaptec3 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_adaptec4.csv ./congestion.csv
./route --input=adaptec4 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_adaptec5.csv ./congestion.csv
./route --input=adaptec5 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_bigblue1.csv ./congestion.csv
./route --input=bigblue1 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_bigblue2.csv ./congestion.csv
./route --input=bigblue2 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_bigblue3.csv ./congestion.csv
./route --input=bigblue3 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_newblue1.csv ./congestion.csv
./route --input=newblue1 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_newblue2.csv ./congestion.csv
./route --input=newblue2 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_newblue5.csv ./congestion.csv
./route --input=newblue5 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_newblue6.csv ./congestion.csv
./route --input=newblue6 --output=output
rm ./congestion.csv


#Hard to Route
cp ./predicted_imgs/congestion_bigblue4.csv ./congestion.csv
./route --input=bigblue4 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_newblue4.csv ./congestion.csv
./route --input=newblue4 --output=output
rm ./congestion.csv

cp ./predicted_imgs/congestion_newblue3.csv ./congestion.csv
./route --input=newblue3 --output=output
rm ./congestion.csv
