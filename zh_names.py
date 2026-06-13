"""
Chinese name mapping for all FIFA World Cup teams.
Maps English team names to standard Simplified Chinese names.
"""
from typing import Dict

TEAM_ZH: Dict[str, str] = {
    # ── 2026 Qualified Teams (48) ──
    'Mexico': '墨西哥',
    'Czech Republic': '捷克',
    'South Africa': '南非',
    'Korea Republic': '韩国',
    'Canada': '加拿大',
    'Bosnia and Herzegovina': '波黑',
    'Qatar': '卡塔尔',
    'Switzerland': '瑞士',
    'Brazil': '巴西',
    'Haiti': '海地',
    'Morocco': '摩洛哥',
    'Scotland': '苏格兰',
    'United States': '美国',
    'Australia': '澳大利亚',
    'Paraguay': '巴拉圭',
    'Turkey': '土耳其',
    'Curacao': '库拉索',
    'Ecuador': '厄瓜多尔',
    'Germany': '德国',
    "Cote d'Ivoire": '科特迪瓦',
    'Netherlands': '荷兰',
    'Japan': '日本',
    'Sweden': '瑞典',
    'Tunisia': '突尼斯',
    'Belgium': '比利时',
    'Egypt': '埃及',
    'Iran': '伊朗',
    'New Zealand': '新西兰',
    'Cape Verde': '佛得角',
    'Saudi Arabia': '沙特阿拉伯',
    'Spain': '西班牙',
    'Uruguay': '乌拉圭',
    'France': '法国',
    'Norway': '挪威',
    'Senegal': '塞内加尔',
    'Iraq': '伊拉克',
    'Algeria': '阿尔及利亚',
    'Argentina': '阿根廷',
    'Austria': '奥地利',
    'Jordan': '约旦',
    'Colombia': '哥伦比亚',
    'DR Congo': '刚果民主共和国',
    'Portugal': '葡萄牙',
    'Uzbekistan': '乌兹别克斯坦',
    'Croatia': '克罗地亚',
    'England': '英格兰',
    'Ghana': '加纳',
    'Panama': '巴拿马',

    # ── Additional Historical Teams ──
    'Italy': '意大利',
    'Denmark': '丹麦',
    'Poland': '波兰',
    'Chile': '智利',
    'Nigeria': '尼日利亚',
    'Cameroon': '喀麦隆',
    'Peru': '秘鲁',
    'Russia': '俄罗斯',
    'Serbia': '塞尔维亚',
    'Greece': '希腊',
    'Ukraine': '乌克兰',
    'Wales': '威尔士',
    'Slovakia': '斯洛伐克',
    'Slovenia': '斯洛文尼亚',
    'Romania': '罗马尼亚',
    'Bulgaria': '保加利亚',
    'Hungary': '匈牙利',
    'Iceland': '冰岛',
    'Costa Rica': '哥斯达黎加',
    'Honduras': '洪都拉斯',
    'Jamaica': '牙买加',
    'El Salvador': '萨尔瓦多',
    'Trinidad and Tobago': '特立尼达和多巴哥',
    'Cuba': '古巴',
    'Bolivia': '玻利维亚',
    'Venezuela': '委内瑞拉',
    'China PR': '中国',
    'Korea DPR': '朝鲜',
    'India': '印度',
    'Kuwait': '科威特',
    'United Arab Emirates': '阿联酋',
    'Israel': '以色列',
    'Togo': '多哥',
    'Angola': '安哥拉',
    'Zaire': '扎伊尔',
    'Northern Ireland': '北爱尔兰',
    'Republic of Ireland': '爱尔兰',
    'Türkiye': '土耳其',

    # ── Defunct/Historical (mapped to successors in data.py) ──
    'Czechoslovakia': '捷克斯洛伐克',
    'West Germany': '西德',
    'East Germany': '东德',
    'Germany DR': '东德',
    'Soviet Union': '苏联',
    'Yugoslavia': '南斯拉夫',
    'FR Yugoslavia': '南斯拉夫联盟',
    'Serbia and Montenegro': '塞尔维亚和黑山',
    'Dutch East Indies': '荷属东印度',

    # ── Other teams in data ──
    'IR Iran': '伊朗',
    'Côte d\'Ivoire': '科特迪瓦',
}


def to_zh(name: str) -> str:
    """Convert an English team name to Chinese. Falls back to English."""
    return TEAM_ZH.get(name, name)


def to_zh_safe(name: str) -> str:
    """Convert to Chinese, with a short fallback (first word of English)."""
    if name in TEAM_ZH:
        return TEAM_ZH[name]
    return name.split(',')[0].strip()
