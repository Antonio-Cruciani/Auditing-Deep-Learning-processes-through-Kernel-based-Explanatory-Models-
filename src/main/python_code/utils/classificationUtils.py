import gzip
#This function writes headers for LRP results.
def prep_sprdsheet(worksheet,dir):

    worksheet.write(0,0,'Class')
    worksheet.write(0,1,'Landmarks')

    testinstances = []
    with gzip.open(dir+'test_ids_class.txt.gz','rt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split('\t')]
            testinstances.append(inner_list)

    counter = 2

    for instance in testinstances:
        stringa = "[" +instance[0].strip() +"]"+ "\t"+instance[1]
        worksheet.write(0,counter,stringa)
        counter +=1

    landmarks = []
    with gzip.open(dir+'landmarks.txt.gz','rt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split('\t')]
            landmarks.append(inner_list)

    counter = 1
    max_len = 0
    for landmark in landmarks:
        if len(landmark[1]) > max_len:
            max_len = len(landmark[1])
             #print(max_len)
            worksheet.set_column(1,1,min(max_len,60))
        worksheet.write(counter,0,landmark[0])
        worksheet.write(counter, 1, landmark[1])
        counter += 1
    return