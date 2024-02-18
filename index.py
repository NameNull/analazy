

# 创建数据集
import json
with open("business_plan_dataset.json", "w") as f:
    json.dump(business_plan_data, f)

print("商业计划书已转换为数据集并保存到 business_plan_dataset.json 文件中。")
