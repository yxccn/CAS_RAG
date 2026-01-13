import uuid

generated_ids = set()

def generate_unique_six_digit_id():
    while True:
        unique_id = uuid.uuid4()
        six_digit_id = unique_id.int % 1000000
        six_digit_str = str(six_digit_id).zfill(6)

        if six_digit_str not in generated_ids:  # 检查唯一性
            generated_ids.add(six_digit_str)
            return six_digit_str

# 生成220个唯一的6位数 ID
for _ in range(220):
    print(generate_unique_six_digit_id())
