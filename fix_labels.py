from pathlib import Path

root = Path(r"C:\Users\chuki\Desktop\jscanner_corner_dataset\v2")

fixed = 0

for txt in root.rglob("*.txt"):
    try:
        content = txt.read_text(encoding="utf-8")

        new_content = content.replace("\\n", "\n")

        if content != new_content:
            txt.write_text(new_content, encoding="utf-8")
            fixed += 1
            print("FIX:", txt.name)

    except Exception as e:
        print("ERR:", txt, e)

print()
print("완료")
print("수정파일:", fixed)