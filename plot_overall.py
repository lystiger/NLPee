import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_full_comparison():
    # 1. Đọc dữ liệu
    df = pd.read_csv('results_log.csv')
    
    # 2. Chuyển đổi sang số (đề phòng dữ liệu bị dính text)
    cols = ['BLEU_Before', 'BLEU_After', 'ROUGE-L']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Tính trung bình theo từng Backend
    summary = df.groupby('Backend')[cols].mean().reset_index()
    
    # 4. "Melt" dữ liệu để Seaborn hiểu được cách vẽ cột nhóm
    df_plot = summary.melt(id_vars='Backend', var_name='Metric', value_name='Score')

    # 5. Thiết lập đồ họa
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Vẽ biểu đồ cột
    ax = sns.barplot(data=df_plot, x='Metric', y='Score', hue='Backend', palette='magma')

    # 6. Trang trí biểu đồ cho "oai" để dán vào báo cáo
    plt.title('SO SÁNH CHI TIẾT HIỆU NĂNG HỆ THỐNG\n(Trước NLP vs Sau NLP vs ROUGE-L)', 
              fontweight='bold', fontsize=15, pad=20)
    plt.ylabel('Điểm trung bình (%)', fontweight='bold', fontsize=12)
    plt.xlabel('Các tiêu chí đánh giá', fontweight='bold', fontsize=12)
    plt.ylim(0, 115)
    
    # Hiển thị số trên đầu cột
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.1f}%', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 9), 
                        textcoords='offset points', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('Full_Metric_Comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    draw_full_comparison()