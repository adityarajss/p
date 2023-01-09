import csv
hypo=[]
data=[]
temp=[]
gen=['?','?','?','?','?','?']
sef=[]
with open('3.Candidate Elimination.csv') as csv_file:
    fd = csv.reader(csv_file)
    print("\nThe given training examples are:")
    for line in fd:
        print(line)
        temp.append(line)
        if line[-1]== "Yes":
            data.append(line)
print("\nThe positive examples are: Enjoy swimming");
for line in data:
    print(line)
row= len(data)
col=len(data[0])
print("\nThe final specific output......................");
for j in range(col-1):
    hypo.append(data[0][j])
for i in range(row):
    for j in range(col-1):
        if (hypo[j]!=data[i][j]):
            hypo[j]='?'
print(hypo)
print("\nThe final Genralize output..................");
row=len(temp)
col=len(temp)
for i in range(row):
    if temp[i][-1]=="No":
        for j in range(col-1):
            if temp[i][j] !=hypo[j]:
                gen[j]=hypo[j]
                print(gen)
                gen[j]='?'