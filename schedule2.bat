set list=PSD SVM CNN CNN_LSTM
set list2=EA DEA False
python train.py --windowSize 128 --modelName PSD --bandL 0.1 --bandR 50 --extractFixation False --thinking True --trainTestSeperate True --trainTestSession False --windowSize 128 --input C:\Users\hmi\Desktop\DataVIN --eaNorm 1EA 
python train.py --windowSize 128 --modelName PSD --bandL 0.1 --bandR 8 --extractFixation False --thinking True --trainTestSeperate True --trainTestSession False --windowSize 128 --input C:\Users\hmi\Desktop\DataVIN --eaNorm 1EA 
python train.py --windowSize 128 --modelName PSD --bandL 8 --bandR 13 --extractFixation False --thinking True --trainTestSeperate True --trainTestSession False --windowSize 128 --input C:\Users\hmi\Desktop\DataVIN --eaNorm 1EA 
python train.py --windowSize 128 --modelName PSD --bandL 13 --bandR 30 --extractFixation False --thinking True --trainTestSeperate True --trainTestSession False --windowSize 128 --input C:\Users\hmi\Desktop\DataVIN --eaNorm 1EA 
python train.py --windowSize 128 --modelName IHAR --bandL 0.1 --bandR 50 --extractFixation False --thinking True --trainTestSeperate True --trainTestSession False --windowSize 128 --input C:\Users\hmi\Desktop\DataVIN --eaNorm 1EA 
python train.py --windowSize 128 --modelName IHAR --bandL 0.1 --bandR 8 --extractFixation False --thinking True --trainTestSeperate True --trainTestSession False --windowSize 128 --input C:\Users\hmi\Desktop\DataVIN --eaNorm 1EA 
python train.py --windowSize 128 --modelName IHAR --bandL 8 --bandR 13 --extractFixation False --thinking True --trainTestSeperate True --trainTestSession False --windowSize 128 --input C:\Users\hmi\Desktop\DataVIN --eaNorm 1EA 
python train.py --windowSize 128 --modelName IHAR --bandL 13 --bandR 30 --extractFixation False --thinking True --trainTestSeperate True --trainTestSession False --windowSize 128 --input C:\Users\hmi\Desktop\DataVIN --eaNorm 1EA 
