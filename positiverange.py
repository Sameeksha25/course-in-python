
li=[]
n=int(input("Enter size of list "))
print("Enter element of list ")
for i in range(0,n):
    e= int(input())
    li.append(e)

print("Positive numbers in",li,"are: ")

for i in li:   
    if i >= 0:
       print(i, " ")
