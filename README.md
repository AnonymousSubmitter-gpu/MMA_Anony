# MMA: Benchmarking Multi-Modal Large Language Models in Ambiguity Contexts

1. You have to set GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY to use this limited-access models.
2. Place your csv file in a pth
3. Download dataset from: https://drive.google.com/file/d/1HP4ldhWMFmr3_Xnh4a97bSlJ-lB6X8kB/view?usp=sharing
4. Place your dataset in a path
5. Install dependency with requirement files.
6. python main.py --model [Model Name] #modify the dataset path and csv path in line#46 #47
7. Find your output csv path after running main.py
8. python evaluation.py  #modify the dataset path and csv path in line#4, and modify the models names in line#7
9. Final Ambiguity Accuracy of model will be given as this order: "Adjective","Noun","Verb", "Attachment Ambiguity", "Coordination Ambiguity","Structural Ambiguity","Pragmatic Ambiguity","Idiom", "Lexical", "Syntactic", "Semantic", "Overall".
