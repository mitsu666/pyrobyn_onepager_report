#!/usr/bin/env python
# coding: utf-8

# In[1]:


# MIT License

# Copyright (c) Meta Platforms, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import japanize_matplotlib
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.nonparametric.smoothers_lowess import lowess

# 1. Spend x effect share comparison
def plot_spend_effect_share(plotMediaShareLoopBar, plotMediaShareLoopLine, ySecScale, dep_var_type, output_filename=None):
    """
    プロット関数: 総支出シェアと効果シェアの比較

    Args:
    - plotMediaShareLoopBar (DataFrame): バーチャート用のメディアシェアデータ。
    - plotMediaShareLoopLine (DataFrame): ラインチャート用のメディア効果データ。
    - ySecScale (float): セカンダリ軸のスケーリング係数。
    - dep_var_type (str): 依存変数のタイプ ('revenue' or 'cpa')。
    - output_filename (str): ファイル名 (JPG形式) を指定すると、プロットを保存します。

    Returns:
    - None: Displays the plot and optionally saves it.
    """ 
    import matplotlib.ticker as ticker

    # タイトルとテキストラベルの設定
    if dep_var_type == 'revenue':
        line_label = "ROI"
        sort_order = plotMediaShareLoopLine.sort_values('value', ascending=False)  # ROIの降順
    else:
        line_label = "CPA"
        sort_order = plotMediaShareLoopLine.sort_values('value', ascending=True)  # CPAの昇順

    plotMediaShareLoopLine = plotMediaShareLoopLine.loc[sort_order.index,:]
    temp1 = plotMediaShareLoopBar[plotMediaShareLoopBar['variable']=="spend_share"].reset_index(drop=True).loc[sort_order.index,:]
    temp2 = plotMediaShareLoopBar[plotMediaShareLoopBar['variable']=="effect_share"].reset_index(drop=True).loc[sort_order.index,:]
    plotMediaShareLoopBar = pd.concat([temp1,temp2])
    # プロット作成
    plt.figure(figsize=(16, 10))
    palette = sns.color_palette("pastel", len(plotMediaShareLoopBar['variable'].unique()))
    ax = sns.barplot(y='rn', x='value', hue='variable', data=plotMediaShareLoopBar, orient='h', palette=palette)

    # 棒グラフ内に%を表示
    for container in ax.containers:
        for bar, label in zip(container, container.datavalues):
            bar_width = bar.get_width()
            plt.text(bar_width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{bar_width * 100:.1f}%', ha='left', va='center', fontsize=12)

    # 線とラベルをプロット
    line_color = "darkblue"
    plt.plot(plotMediaShareLoopLine['value'] / ySecScale, plotMediaShareLoopLine['rn'], 'o-', 
             color=line_color, linewidth=1.5, label=line_label)
    for x, y, value in zip(plotMediaShareLoopLine['value'] / ySecScale, plotMediaShareLoopLine['rn'], plotMediaShareLoopLine['value']):
        plt.text(x, y, f"{round(value, 2)}", color=line_color, ha='left', va='center', fontsize=20, weight='bold')

    # プロットのスタイル設定
    plt.title("Share of Total Spend, Effect & Type in Modeling Window", fontsize=20, weight='bold')
    plt.xlabel("Percentage (%)", fontsize=20)
    plt.legend(title=None, loc='lower right', fontsize=20, frameon=True, fancybox=True, shadow=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%'))
    plt.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.5)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    # ファイル保存オプション
    if output_filename:
        plt.savefig(output_filename, format='jpg', dpi=300)
    plt.show()

# 2. Waterfall plot
def plot_waterfall(plotWaterfallLoop, output_filename=None):
    """
    プロット関数: ウォーターフォールプロット (寄与分解)

    Args:
    - plotWaterfallLoop (DataFrame): 各寄与成分のデータ（正負を含む）。
    - output_filename (str): ファイル名 (JPG形式) を指定すると、プロットを保存します。

    Returns:
    - None: Displays the plot and optionally saves it.
    """
    plt.figure(figsize=(16, 10))
    plotWaterfallLoop['start'] = plotWaterfallLoop['end'].shift(1).fillna(1)
    plotWaterfallLoop['y_pos'] = (plotWaterfallLoop['start'] + plotWaterfallLoop['end']) / 2
    colors = {"Positive": "#59B3D2", "Negative": "#E5586E"}
    for _, row in plotWaterfallLoop.iloc[::-1].iterrows():
        plt.barh(row['rn'], width=row['end'] - row['start'], left=row['start'], color=colors[row['sign']])
        plt.text(row['y_pos'], row['rn'], f"{round(row['xDecompAgg'], 2)}\n{round(row['xDecompPerc']*100, 1)}%", ha='center', fontsize=16)
    plt.xlabel("Percentage (%)", fontsize=20)
    plt.title("Response Decomposition Waterfall by Predictor", fontsize=20)
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%'))
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, format='jpg', dpi=300)
    plt.show()

# 3. Adstock rate (Geometric or Weibull)
def plot_adstock(dt_geometric, weibullCollect=None, adstock_type="geometric", output_filename=None):
    """
    プロット関数: 広告アドストック率の可視化 (幾何学的またはワイブル)

    Args:
    - dt_geometric (DataFrame): 幾何学的アドストックのデータ。
    - weibullCollect (DataFrame, optional): ワイブルアドストックのデータ。
    - adstock_type (str): アドストックタイプ ('geometric', 'weibull_cdf', 'weibull_pdf')。
    - output_filename (str): ファイル名 (JPG形式) を指定すると、プロットを保存します。

    Returns:
    - None: Displays the plot and optionally saves it.
    """
    plt.figure(figsize=(16, 10))
    if adstock_type == "geometric":
        sns.barplot(y='channels', x='thetas', data=dt_geometric, color="coral", orient='h')
        plt.xlabel("Thetas (%)", fontsize=20)
        plt.ylabel("Channels", fontsize=20)
        plt.title("Geometric Adstock: Fixed Rate Over Time", fontsize=20)
        plt.xlim(0, 1)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%'))
        for index, row in dt_geometric.iterrows():
            plt.text(row['thetas'] + 0.01, index, f"{round(row['thetas'] * 100, 1)}%", color="black", ha="center", va="center", fontsize=15, weight='bold')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    elif adstock_type in ["weibull_cdf", "weibull_pdf"]:
        for channel in weibullCollect['channel'].unique():
            channel_data = weibullCollect[weibullCollect['channel'] == channel]
            plt.plot(channel_data['x'], channel_data['decay_accumulated'], label=channel)
            plt.axhline(0.5, color="gray", linestyle="--")
            plt.text(channel_data['x'].max(), 0.5, "Halflife", va='bottom', ha='right', color="gray", fontsize=20)
        plt.xlabel(f"Time unit [{adstock_type}]", fontsize=20)
        plt.ylabel("Decay Accumulated", fontsize=20)
        plt.title(f"Weibull {adstock_type} Adstock: Flexible Rate Over Time", fontsize=20)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, format='jpg', dpi=300)
    plt.show()

# 4. Response curves
def plot_response_curves(dt_scurvePlot, dt_scurvePlotMean, trim_rate=1.3, output_filename=None):
    """
    プロット関数: 各メディアチャンネルのレスポンス曲線

    Args:
    - dt_scurvePlot (DataFrame): レスポンス曲線のデータ。
    - dt_scurvePlotMean (DataFrame): レスポンス曲線の平均データ。
    - trim_rate (float): グラフのトリムレート。
    - output_filename (str): ファイル名 (JPG形式) を指定すると、プロットを保存します。

    Returns:
    - None: Displays the plot and optionally saves it.
    """
    max_spend = dt_scurvePlotMean['mean_spend_adstocked'].max() * trim_rate
    max_response = dt_scurvePlotMean['mean_response'].max() * trim_rate
    filtered_dt_scurvePlot = dt_scurvePlot[
        (dt_scurvePlot['spend'] < max_spend) &
        (dt_scurvePlot['response'] < max_response) &
        (dt_scurvePlot['channel'].isin(dt_scurvePlotMean['channel']))
    ]
    dt_scurvePlotMean = dt_scurvePlotMean.rename(columns={"rn": "channel"})
    filtered_dt_scurvePlot = filtered_dt_scurvePlot.merge(
        dt_scurvePlotMean[['channel', 'mean_carryover']],
        on='channel', how='left'
    )
    plt.figure(figsize=(16, 10))
    for channel in filtered_dt_scurvePlot['channel'].unique():
        channel_data = filtered_dt_scurvePlot[filtered_dt_scurvePlot['channel'] == channel].sort_values(by='spend')
        plt.plot(channel_data['spend'], channel_data['response'], label=channel)
        carryover_data = channel_data[channel_data['spend'] <= channel_data['mean_carryover']]
        plt.fill_between(carryover_data['spend'], carryover_data['response'], color="grey", alpha=0.4)
    plt.scatter(dt_scurvePlotMean['mean_spend_adstocked'], dt_scurvePlotMean['mean_response'], color='black')
    for i, row in dt_scurvePlotMean.iterrows():
        plt.text(row['mean_spend_adstocked'], row['mean_response'], f"{round(row['mean_spend_adstocked'], 2)}",
                 ha='left', va='center', fontsize=20)
    plt.title("Response Curves and Mean Spends by Channel", fontsize=20)
    plt.xlabel("Spend (carryover + immediate)", fontsize=20)
    plt.ylabel("Response", fontsize=20)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x / 1000)}k'))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='lower right', fontsize=20, frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, format='jpg', dpi=300)
    plt.show()

# 5. Fitted vs actual
def plot_fitted_vs_actual(xDecompVecPlotMelted, train_size, output_filename=None):
    """
    プロット関数: 実際のレスポンス vs 予測レスポンス

    Args:
    - xDecompVecPlotMelted (DataFrame): Melted data containing actual and predicted values.
    - train_size (float): Proportion of the data used for training.
    - output_filename (str): ファイル名 (JPG形式) を指定すると、プロットを保存します。

    Returns:
    - None: Displays the plot and optionally saves it.
    """
    xDecompVecPlotMelted['ds'] = pd.to_datetime(xDecompVecPlotMelted['ds'])
    plt.figure(figsize=(16, 10))
    sns.lineplot(x='ds', y='value', hue='variable', data=xDecompVecPlotMelted, linewidth=2.5)
    
    days = sorted(xDecompVecPlotMelted['ds'].unique())
    total_days = len(days)
    train_cut = int(total_days * train_size)
    val_cut = train_cut + int(total_days * (1 - train_size) / 2)

    plt.axvline(days[train_cut], color="#39638b", alpha=0.8, linewidth=1.5)
    plt.text(days[train_cut], plt.gca().get_ylim()[1], f"Train: {train_size * 100:.1f}%", rotation=270, verticalalignment='bottom', color="#39638b", alpha=0.5, fontsize=20)
    plt.axvline(days[val_cut], color="#39638b", alpha=0.8, linewidth=1.5)
    plt.text(days[val_cut], plt.gca().get_ylim()[1], f"Validation: {(1 - train_size) / 2 * 100:.1f}%", rotation=270, verticalalignment='bottom', color="#39638b", alpha=0.5, fontsize=20)
    plt.axvline(days[-1], color="#39638b", alpha=0.8, linewidth=1.5)
    plt.text(days[-1], plt.gca().get_ylim()[1], f"Test: {(1 - train_size) / 2 * 100:.1f}%", rotation=270, verticalalignment='bottom', color="#39638b", alpha=0.5, fontsize=20)

    plt.title("Actual vs. Predicted Response", fontsize=20)
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Response", fontsize=20)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    plt.legend(title=None, loc='upper left', fontsize=20)
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, format='jpg', dpi=300)
    plt.show()

# 6. Diagnostic: fitted vs residual
def plot_diagnostic(xDecompVecPlot, output_filename=None):
    """
    プロット関数: フィッティング結果 vs 残差

    Args:
    - xDecompVecPlot (DataFrame): Data containing fitted and actual values.
    - output_filename (str): ファイル名 (JPG形式) を指定すると、プロットを保存します。

    Returns:
    - None: Displays the plot and optionally saves it.
    """
    xDecompVecPlot['actual'] = pd.to_numeric(xDecompVecPlot['actual'], errors='coerce')
    xDecompVecPlot['predicted'] = pd.to_numeric(xDecompVecPlot['predicted'], errors='coerce')
    xDecompVecPlot['residual'] = xDecompVecPlot['actual'] - xDecompVecPlot['predicted']
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x='predicted', y='residual', data=xDecompVecPlot)
    plt.axhline(y=0, color='black', linestyle='--')
    
    loess_smooth = lowess(xDecompVecPlot['residual'], xDecompVecPlot['predicted'], frac=0.3)
    plt.plot(loess_smooth[:, 0], loess_smooth[:, 1], color='blue')
    
    plt.xlabel("Fitted", fontsize=20)
    plt.ylabel("Residual", fontsize=20)
    plt.title("Fitted vs. Residual", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, format='jpg', dpi=300)
    plt.show()

# 7. Immediate vs carryover
def plot_immediate_vs_carryover(df_imme_caov, output_filename=None):
    """
    プロット関数: 即時効果 vs キャリーオーバー効果

    Args:
    - df_imme_caov (DataFrame): Data containing immediate and carryover percentages.
    - output_filename (str): ファイル名 (JPG形式) を指定すると、プロットを保存します。

    Returns:
    - None: Displays the plot and optionally saves it.
    """
    plt.figure(figsize=(16, 10))
    immediate = df_imme_caov[df_imme_caov['type'] == 'Immediate']
    carryover = df_imme_caov[df_imme_caov['type'] == 'Carryover']
    
    plt.barh(immediate['rn'], immediate['percentage'], color='#59B3D2', label='Immediate')
    plt.barh(carryover['rn'], carryover['percentage'], left=immediate['percentage'], color='coral', label='Carryover')
    
    for index, row in immediate.iterrows():
        plt.text(row['percentage'] / 2, row['rn'], f"{row['percentage']*100:.1f}%", ha='center', va='center', fontsize=20, color='white', weight='bold')
    for index, row in carryover.iterrows():
        plt.text(row['percentage'] / 2 + immediate[immediate['rn'] == row['rn']]['percentage'].values[0], row['rn'], f"{row['percentage']*100:.1f}%", ha='center', va='center', fontsize=20, color='white', weight='bold')

    plt.xlabel("Percentage (%)", fontsize=20)
    plt.title("Immediate vs. Carryover Response Percentage", fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.xlim(0, 1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%'))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, format='jpg', dpi=300)
    plt.show()
