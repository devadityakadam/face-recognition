import mysql.connector
mydb=mysql.connector.connect(host="localhost",user="root",passwd="pass@123",database="pro_student")
mycursor=mydb.cursor()
mycursor.execute("create table authorize(id int auto_increment primary key,name varchar(50),age int,address varchar(50))")
mycursor.execute("show tables;")
for x in mycursor:
    print(x)
#sql="insert into stu_table(id,name,age)values(%s,%s,%s)"
#val=(2,"Nehal Meskar",23)
#mycursor.execute(sql,val)
#mydb.commit()
#print(mycursor.rowcount,"record inserted")

#mycursor.execute("select * from stu_table")
#myresult=mycursor.fetchall()
#for x in myresult:
#    print(x)