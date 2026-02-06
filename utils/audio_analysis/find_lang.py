import json
import os
from pathlib import Path

meta_file = ['/apdcephfs_jn2/share_302243908/terohu/data/human-avideo/openhumanvid0812.list',
             '/apdcephfs_jn2/share_302243908/terohu/data/human-avideo/talking_head0808.list',
             '/apdcephfs_jn2/share_302243908/terohu/data/human-avideo/openhumanvid_140w_caped_0827.list']

def process_list_files():
    """
    读取每个list文件中的JSON数据，根据lang字段分类写入新文件
    """
    for file_path in meta_file:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        # 获取文件名（不含扩展名）
        file_name = Path(file_path).stem
        file_dir = Path(file_path).parent
        
        # 创建输出文件路径
        en_output = file_dir / f"{file_name}_en.list"
        zh_output = file_dir / f"{file_name}_zh.list"
        
        # 初始化计数器
        en_count = 0
        zh_count = 0
        total_count = 0
        
        print(f"正在处理文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                with open(en_output, 'w', encoding='utf-8') as en_file, \
                     open(zh_output, 'w', encoding='utf-8') as zh_file:
                    
                    for line_num, line in enumerate(f, 1):
                        if line_num % 10000 == 0:
                            print(f"已处理 {line_num} 行...")
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            # 解析JSON
                            data=json.load(open(line, "r"))
                            total_count += 1
                            
                            # 检查lang字段
                            if 'language' in data:
                                lang = data['language'].lower()
                                if lang == 'en':
                                    en_file.write(line + '\n')
                                    en_count += 1
                                elif lang == 'zh':
                                    zh_file.write(line + '\n')
                                    zh_count += 1
                                else:
                                    print(f"未知语言类型 '{lang}' 在第 {line_num} 行: {file_path}")
                            else:
                                print(f"缺少 'language' 字段在第 {line_num} 行: {file_path}")
                                
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误在第 {line_num} 行: {file_path}, 错误: {e}")
                            continue
                            
        except Exception as e:
            print(f"处理文件时出错 {file_path}: {e}")
            continue
            
        print(f"完成处理 {file_path}:")
        print(f"  总计: {total_count} 条记录")
        print(f"  英文: {en_count} 条记录 -> {en_output}")
        print(f"  中文: {zh_count} 条记录 -> {zh_output}")
        print("-" * 50)

if __name__ == "__main__":
    process_list_files()
