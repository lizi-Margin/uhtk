#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准大气模型模拟 - 声速与高度关系图
Standard Atmosphere Model - Speed of Sound vs Altitude
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    warnings.filterwarnings('default')  # Show all warnings

    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")

    # Configure platform-appropriate fonts for cross-platform compatibility
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

def standard_atmosphere_model(h_ft):
    """
    标准大气模型计算
    International Standard Atmosphere (ISA) model
    
    Parameters:
    h_ft: 高度 (英尺)
    
    Returns:
    dict: 包含气压、温度、密度、聲速等参数
    """
    # 标准大气参数
    T0 = 288.15  # 海平面标准温度 (K)
    p0 = 101325  # 海平面标准气压 (Pa)
    rho0 = 1.225  # 海平面标准密度 (kg/m³)
    g = 9.80665  # 重力加速度 (m/s²)
    R = 287.05  # 空气气体常数 (J/kg·K)
    gamma = 1.4  # 比热比
    
    # 转换为米
    h_m = h_ft * 0.3048
    
    # 分层大气模型
    if h_m <= 11000:  # 对流层
        T = T0 - 0.0065 * h_m
        p = p0 * (1 - 0.0065 * h_m / T0) ** (g / (R * 0.0065))
    elif h_m <= 20000:  # 平流层下部 (等温层)
        T = 216.65  # K
        p = 22632 * np.exp(-g * (h_m - 11000) / (R * T))
    else:  # 平流层上部 (超出范围)
        T = 216.65  # K
        p = 5474.9 * (216.65 / 216.65) ** (g / (R * 0.001))  # 简化处理
    
    # 计算密度
    rho = p / (R * T)
    
    # 计算声速 (m/s)
    speed_of_sound = np.sqrt(gamma * R * T)
    
    # 计算等效表速 (KTAS) - True Air Speed in knots
    # EAS = SQRT(2 * q / rho_air), 其中 q = 1/2 * rho * V²
    # 简化计算：表速与真速的关系
    true_speed_ms = speed_of_sound  # 使用声速作为参考
    true_speed_kt = true_speed_ms * 1.94384  # 转换为节
    
    # 等效表速 (EAS) 计算
    # EAS = TAS * SQRT(rho / rho0)
    equivalent_airspeed_kt = true_speed_kt * np.sqrt(rho / rho0)
    
    return {
        'altitude_ft': h_ft,
        'altitude_m': h_m,
        'temperature_K': T,
        'temperature_C': T - 273.15,
        'pressure_Pa': p,
        'pressure_hPa': p / 100,
        'density_kg_m3': rho,
        'speed_of_sound_ms': speed_of_sound,
        'speed_of_sound_kt': speed_of_sound * 1.94384,
        'equivalent_airspeed_kt': equivalent_airspeed_kt,
        'true_airspeed_kt': true_speed_kt
    }

def create_atmospheric_plots():
    """
    创建大气模型图表
    """
    setup_matplotlib_for_plotting()
    
    # 生成高度数据 (0-50000 ft)
    altitudes_ft = np.linspace(0, 50000, 1000)
    
    # 计算大气参数
    atmospheric_data = []
    for alt in altitudes_ft:
        data = standard_atmosphere_model(alt)
        atmospheric_data.append(data)
    
    # 转换为numpy数组便于绘图
    altitudes = np.array([d['altitude_ft'] for d in atmospheric_data])
    temperatures = np.array([d['temperature_C'] for d in atmospheric_data])
    pressures = np.array([d['pressure_hPa'] for d in atmospheric_data])
    densities = np.array([d['density_kg_m3'] for d in atmospheric_data])
    speed_of_sound = np.array([d['speed_of_sound_ms'] for d in atmospheric_data])
    e_airspeed = np.array([d['equivalent_airspeed_kt'] for d in atmospheric_data])
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('标准大气模型模拟 (0-50,000 ft)\nStandard Atmosphere Model Simulation', fontsize=16, fontweight='bold')
    
    # 1. 高度-声速图
    ax1.plot(speed_of_sound, altitudes, 'b-', linewidth=2, label='声速 / Speed of Sound')
    ax1.set_xlabel('声速 (m/s) / Speed of Sound (m/s)')
    ax1.set_ylabel('高度 (ft) / Altitude (ft)')
    ax1.set_title('高度-声速关系图\nAltitude vs Speed of Sound')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # 高度从下到上递增
    ax1.legend()
    
    # 2. 高度-表速图
    ax2.plot(e_airspeed, altitudes, 'r-', linewidth=2, label='等效表速 / Equivalent Airspeed')
    ax2.set_xlabel('表速 (节) / Airspeed (knots)')
    ax2.set_ylabel('高度 (ft) / Altitude (ft)')
    ax2.set_title('高度-表速关系图\nAltitude vs Equivalent Airspeed')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    ax2.legend()
    
    # 3. 高度-温度图
    ax3.plot(temperatures, altitudes, 'orange', linewidth=2, label='温度 / Temperature')
    ax3.set_xlabel('温度 (°C) / Temperature (°C)')
    ax3.set_ylabel('高度 (ft) / Altitude (ft)')
    ax3.set_title('高度-温度关系图\nAltitude vs Temperature')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    ax3.legend()
    
    # 4. 高度-密度图
    ax4.plot(densities, altitudes, 'g-', linewidth=2, label='密度 / Density')
    ax4.set_xlabel('密度 (kg/m³) / Density (kg/m³)')
    ax4.set_ylabel('高度 (ft) / Altitude (ft)')
    ax4.set_title('高度-密度关系图\nAltitude vs Density')
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('/workspace/atmospheric_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建单独的高度-声速图 (更大的版本)
    plt.figure(figsize=(10, 12))
    plt.plot(speed_of_sound, altitudes, 'b-', linewidth=3, label='声速 / Speed of Sound')
    plt.xlabel('声速 (m/s) / Speed of Sound (m/s)', fontsize=14)
    plt.ylabel('高度 (ft) / Altitude (ft)', fontsize=14)
    plt.title('高度-声速关系图 (0-50,000 ft)\nAltitude vs Speed of Sound', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=12)
    
    # 添加关键点标注
    sea_level_idx = np.argmin(np.abs(altitudes - 0))
    stratosphere_idx = np.argmin(np.abs(altitudes - 36089))  # 对流层顶
    
    plt.annotate(f'海平面 / Sea Level\n{speed_of_sound[sea_level_idx]:.1f} m/s', 
                xy=(speed_of_sound[sea_level_idx], altitudes[sea_level_idx]), 
                xytext=(speed_of_sound[sea_level_idx] + 20, altitudes[sea_level_idx] + 5000),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='left')
    
    plt.annotate(f'对流层顶 / Tropopause\n{speed_of_sound[stratosphere_idx]:.1f} m/s', 
                xy=(speed_of_sound[stratosphere_idx], altitudes[stratosphere_idx]), 
                xytext=(speed_of_sound[stratosphere_idx] + 20, altitudes[stratosphere_idx] - 5000),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.savefig('/workspace/altitude_speed_of_sound.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建单独的高度-表速图 (更大的版本)
    plt.figure(figsize=(10, 12))
    plt.plot(e_airspeed, altitudes, 'r-', linewidth=3, label='等效表速 / Equivalent Airspeed')
    plt.xlabel('表速 (节) / Airspeed (knots)', fontsize=14)
    plt.ylabel('高度 (ft) / Altitude (ft)', fontsize=14)
    plt.title('高度-表速关系图 (0-50,000 ft)\nAltitude vs Equivalent Airspeed', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=12)
    
    # 添加关键点标注
    plt.annotate(f'海平面 / Sea Level\n{e_airspeed[sea_level_idx]:.1f} knots', 
                xy=(e_airspeed[sea_level_idx], altitudes[sea_level_idx]), 
                xytext=(e_airspeed[sea_level_idx] + 20, altitudes[sea_level_idx] + 5000),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, ha='left')
    
    plt.annotate(f'对流层顶 / Tropopause\n{e_airspeed[stratosphere_idx]:.1f} knots', 
                xy=(e_airspeed[stratosphere_idx], altitudes[stratosphere_idx]), 
                xytext=(e_airspeed[stratosphere_idx] + 20, altitudes[stratosphere_idx] - 5000),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.savefig('/workspace/altitude_airspeed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return atmospheric_data

if __name__ == "__main__":
    print("开始计算标准大气模型...")
    data = create_atmospheric_plots()
    
    # 显示关键数据点
    print("\n=== 关键高度数据 ===")
    key_altitudes = [0, 10000, 20000, 36089, 50000]
    
    for alt_ft in key_altitudes:
        idx = np.argmin(np.abs(np.array([d['altitude_ft'] for d in data]) - alt_ft))
        d = data[idx]
        print(f"\n高度: {d['altitude_ft']:,.0f} ft")
        print(f"  温度: {d['temperature_C']:.1f} °C")
        print(f"  气压: {d['pressure_hPa']:.1f} hPa")
        print(f"  密度: {d['density_kg_m3']:.4f} kg/m³")
        print(f"  声速: {d['speed_of_sound_ms']:.1f} m/s ({d['speed_of_sound_kt']:.1f} knots)")
        print(f"  等效表速: {d['equivalent_airspeed_kt']:.1f} knots")
    
    print(f"\n图表已保存到:")
    print(f"- /workspace/atmospheric_model_analysis.png (综合分析图)")
    print(f"- /workspace/altitude_speed_of_sound.png (高度-声速图)")
    print(f"- /workspace/altitude_airspeed.png (高度-表速图)")