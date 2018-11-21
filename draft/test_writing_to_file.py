name = "Kareem"
familyName = "Jeiroudi"
age = 20
f = open("familyName.txt", "a")
line = "Welcome {0} {1}\nYou're {2} years old".format(name, familyName, age)
print(line)
f.write(line)
line = "Is this a new line?"
f.write(line)
f.close()