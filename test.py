import face_recognition as fr

ronaldinho = fr.load_image_file("ronaldinho.jpg")
unknown = fr.load_image_file("unknown.jpeg")
pele = fr.load_image_file("pele.jpg")
woody = fr.load_image_file("woody-allen.jpg")

ronaldinho_encoding = fr.face_encodings(ronaldinho)[0]
unknown_encoding = fr.face_encodings(unknown)[0]
pele_encoding = fr.face_encodings(pele)[0]
woody_encoding = fr.face_encodings(woody)[0]

#result1 and result2 compare faces "one-on-one"
result1 = fr.compare_faces([ronaldinho_encoding], unknown_encoding)
result2 = fr.compare_faces([ronaldinho_encoding], pele_encoding)

#result3 compares ronaldinho, unknown and pele images to woody image individually
result3 = fr.compare_faces([ronaldinho_encoding, unknown_encoding, pele_encoding], woody_encoding)

print(result1)
print(result2)
print(result3)
