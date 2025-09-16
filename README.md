# 电子商务交易数据分析仪表板 (E-commerce Transaction Analysis Dashboard)


这是一个基于 Streamlit 构建的交互式 Web 应用，旨在对电子商务交易数据进行深度分析和可视化。用户可以通过这个仪表板直观地了解销售趋势、客户行为、国家分布和产品表现。

**[>> 点击这里访问在线演示 <<](https://your-app-url.streamlit.app)**

*(注意：请在将应用部署到 Streamlit Community Cloud 后，将上面的 `https://your-app-url.streamlit.app` 替换为您的实际应用链接)*

---

## ✨ 主要功能

*   **📊 数据概览**: 动态展示数据集的关键信息，包括总销售额、总交易量、独立客户数和国家分布。
*   **📈 销售趋势分析**: 按年、月、日、小时等不同时间维度，交互式探索销售额和订单量的变化趋势。
*   **🌍 地理空间分析**: 通过地图和条形图展示不同国家/地区的销售贡献。
*   **👥 客户价值分析**: 查看消费金额最高的 Top 10 客户列表及其贡献。
*   **📦 产品分析**: 展示最畅销的商品，帮助洞察热门产品。

## ⚙️ 技术栈

*   **Python 3.10+**
*   **Streamlit**: 用于构建交互式 Web 应用界面。
*   **Pandas**: 用于数据处理和分析。
*   **Plotly**: 用于生成交互式图表。
*   **Seaborn & Matplotlib**: 用于数据可视化。

## 🚀 如何在本地运行？

如果您想在自己的电脑上运行这个项目，请按照以下步骤操作：

1.  **克隆仓库**
    ```bash
    git clone https://github.com/yqq-jpg/Data_Analysis.git
    cd Data_Analysis
    ```

2.  **创建并激活虚拟环境 (推荐)**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安装依赖项**
    项目所需的所有库都记录在 `requirements.txt` 文件中。运行以下命令进行安装：
    ```bash
    pip install -r requirements.txt
    ```

4.  **运行 Streamlit 应用**
    一切准备就绪后，在终端中运行以下命令：
    ```bash
    streamlit run app.py
    ```
    您的浏览器将自动打开一个新的标签页，加载此数据分析应用。

## ☁️ 如何部署？

本项目已配置为可以轻松部署到 **Streamlit Community Cloud**。

1.  将此仓库上传到您的 GitHub 账户。
2.  注册/登录 [Streamlit Community Cloud](https://share.streamlit.io/) (使用 GitHub 账户授权)。
3.  点击 "New app"，选择此仓库和 `main` 分支。
4.  确认主文件路径为 `app.py`。
5.  点击 "Deploy!"，等待几分钟即可完成部署。

部署成功后，您将获得一个公开的 URL 链接，任何人都可以通过该链接访问您的数据分析仪表板。

## 📂 文件结构

```
.
├── app.py                     # Streamlit 应用主程序
├── ecommerce_transactions.csv # 数据集文件
├── requirements.txt           # Python 依赖库列表
└── README.md                  # 项目说明文件
```
