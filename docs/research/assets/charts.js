// 图表初始化（ECharts + Mermaid）—— 所有颜色取自 CSS 变量
(function () {
  var s = getComputedStyle(document.documentElement);
  var accent = s.getPropertyValue('--accent').trim();
  var accent2 = s.getPropertyValue('--accent2').trim();
  var ink = s.getPropertyValue('--ink').trim();
  var muted = s.getPropertyValue('--muted').trim();
  var rule = s.getPropertyValue('--rule').trim();
  var bg2 = s.getPropertyValue('--bg2').trim();
  var ok = s.getPropertyValue('--ok').trim();

  var BASE = {
    animation: false,
    textStyle: { fontFamily: 'Instrument, sans-serif', color: ink },
    tooltip: { appendToBody: true, backgroundColor: bg2, borderColor: rule, textStyle: { color: ink, fontSize: 12 } },
    grid: { left: 60, right: 30, top: 30, bottom: 48, containLabel: true }
  };

  function mount(id, option) {
    var el = document.getElementById(id);
    if (!el || typeof echarts === 'undefined') return null;
    var c = echarts.init(el, null, { renderer: 'svg' });
    c.setOption(option);
    window.addEventListener('resize', function () { c.resize(); });
    return c;
  }

  // ---------- 图3 显存峰值（实测） ----------
  mount('chart-vram', Object.assign({}, BASE, {
    tooltip: Object.assign({}, BASE.tooltip, { trigger: 'axis', axisPointer: { type: 'shadow' },
      formatter: function (p) { return p[0].name + '<br/>峰值显存：<b>' + p[0].value + ' GB</b>'; } }),
    xAxis: { type: 'category', data: ['全参微调\n(0.5B, 4×512)', 'LoRA\n(0.5B, 4×512)', 'LoRA+梯度检查点\n(0.5B, 2×1024)', '显卡物理上限'],
      axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted, fontSize: 11, lineHeight: 14 } },
    yAxis: { type: 'value', name: 'GB', nameTextStyle: { color: muted }, splitLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    series: [{
      type: 'bar', barWidth: '46%',
      data: [
        { value: 10.26, itemStyle: { color: accent2 } },
        { value: 8.18, itemStyle: { color: accent2 + 'bb' } },
        { value: 3.5, itemStyle: { color: ok } },
        { value: 4.0, itemStyle: { color: muted + '55' } }
      ],
      label: { show: true, position: 'top', color: ink, fontFamily: 'JBMono', fontSize: 12, formatter: '{c} GB' },
      markLine: { silent: true, symbol: 'none', lineStyle: { color: accent2, type: 'dashed', width: 1.5 },
        data: [{ yAxis: 4.0, label: { formatter: '4GB 物理上限', color: accent2, fontSize: 11, position: 'insideEndTop' } }] }
    }]
  }));

  // ---------- 图4 训练吞吐（实测，对数刻度） ----------
  mount('chart-speed', Object.assign({}, BASE, {
    tooltip: Object.assign({}, BASE.tooltip, { trigger: 'axis', axisPointer: { type: 'shadow' },
      formatter: function (p) { return p[0].name + '<br/>每样本：<b>' + p[0].value + ' 秒</b>'; } }),
    xAxis: { type: 'category', data: ['纯 CPU 训练\n(0.5B)', 'GPU LoRA + 梯度检查点\n(0.5B，优化后)', 'GPU 4-bit QLoRA\n(1.5B，仍溢出)'],
      axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted, fontSize: 11, lineHeight: 14 } },
    yAxis: { type: 'log', name: '秒/样本（越低越好，对数轴）', nameTextStyle: { color: muted, fontSize: 11 },
      min: 0.2, max: 20, splitLine: { lineStyle: { color: rule } }, axisLabel: { color: muted, formatter: '{value}' } },
    series: [{
      type: 'bar', barWidth: '46%',
      data: [
        { value: 4.39, itemStyle: { color: accent2 } },
        { value: 0.30, itemStyle: { color: ok } },
        { value: 15.29, itemStyle: { color: accent2 } }
      ],
      label: { show: true, position: 'top', color: ink, fontFamily: 'JBMono', fontSize: 12, formatter: '{c}s' }
    }]
  }));

  // ---------- 图5 损失曲线（实测） ----------
  var lossData = [1.019, 0.6461, 0.5075, 0.4288, 0.3786, 0.3114, 0.308, 0.2322, 0.154, 0.0899,
    0.1808, 0.1071, 0.0685, 0.0576, 0.0329, 0.0224, 0.0602, 0.0288, 0.0605, 0.0217, 0.0725, 0.0286, 0.0554];
  mount('chart-loss', Object.assign({}, BASE, {
    tooltip: Object.assign({}, BASE.tooltip, { trigger: 'axis',
      formatter: function (p) { return '训练进度 ' + p[0].axisValue + '<br/>损失 <b>' + p[0].value + '</b>'; } }),
    xAxis: { type: 'category', name: '训练进度', nameTextStyle: { color: muted },
      data: lossData.map(function (_, i) { return (i + 1) + ''; }),
      axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted, interval: 2 } },
    yAxis: { type: 'value', name: '损失', nameTextStyle: { color: muted }, splitLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    series: [{
      type: 'line', data: lossData, smooth: true, symbol: 'circle', symbolSize: 6,
      lineStyle: { color: accent, width: 2 }, itemStyle: { color: accent },
      areaStyle: { color: accent, opacity: 0.10 },
      markPoint: { symbolSize: 46, data: [
        { type: 'min', name: '最低', label: { color: '#fff', fontSize: 10 } }
      ] }
    }]
  }));

  // ---------- 图6 路线雷达（定性评估） ----------
  mount('chart-routes', Object.assign({}, BASE, {
    tooltip: Object.assign({}, BASE.tooltip, { trigger: 'item' }),
    legend: { bottom: 0, textStyle: { color: muted, fontSize: 11 }, itemWidth: 12, itemHeight: 8 },
    radar: {
      center: ['50%', '46%'], radius: '64%',
      indicator: [
        { name: '显存友好', max: 5 }, { name: '速度', max: 5 }, { name: '质量上限', max: 5 },
        { name: '易落地', max: 5 }, { name: '本机适配', max: 5 }
      ],
      axisName: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule } },
      splitArea: { areaStyle: { color: ['transparent'] } },
      axisLine: { lineStyle: { color: rule } }
    },
    series: [{
      type: 'radar', symbolSize: 4,
      data: [
        { value: [0, 2, 5, 2, 0], name: 'R1 纯堆参数', lineStyle: { color: muted }, itemStyle: { color: muted } },
        { value: [4, 4, 3, 5, 5], name: 'R2 仅量化', lineStyle: { color: accent2 }, itemStyle: { color: accent2 } },
        { value: [4, 5, 4, 4, 5], name: 'R3 分块并行', lineStyle: { color: accent }, itemStyle: { color: accent } },
        { value: [5, 3, 4, 3, 4], name: 'R4 异构卸载', lineStyle: { color: ok }, itemStyle: { color: ok } },
        { value: [5, 3, 2, 1, 1], name: 'R5 类脑稀疏', lineStyle: { color: accent2 + '88', type: 'dashed' }, itemStyle: { color: accent2 + '88' } }
      ]
    }]
  }));

  // ---------- 图7 性能预期（估算区间） ----------
  var cats = ['0.5B', '1B', '1.5B', '3B', '7B'];
  var lo = [25, 12, 6, 3, 1];
  var hi = [45, 25, 14, 8, 3];
  var band = hi.map(function (v, i) { return v - lo[i]; });
  var mid = hi.map(function (v, i) { return (v + lo[i]) / 2; });
  mount('chart-projection', Object.assign({}, BASE, {
    tooltip: Object.assign({}, BASE.tooltip, { trigger: 'axis',
      formatter: function (p) {
        var i = p[0].dataIndex;
        return '模型规模 ' + cats[i] + '<br/>预期解码：<b>' + lo[i] + '–' + hi[i] + ' tok/s</b>';
      } }),
    legend: { bottom: 0, textStyle: { color: muted, fontSize: 11 }, itemWidth: 12, itemHeight: 8 },
    xAxis: { type: 'category', data: cats, name: '模型规模', nameTextStyle: { color: muted },
      axisLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    yAxis: { type: 'value', name: 'tok/s', nameTextStyle: { color: muted }, splitLine: { lineStyle: { color: rule } }, axisLabel: { color: muted } },
    series: [
      { name: '区间下沿', type: 'line', data: lo, stack: 'r', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, silent: true },
      { name: '不确定性区间', type: 'line', data: band, stack: 'r', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: accent, opacity: 0.13 } },
      { name: '预期中值', type: 'line', data: mid, smooth: true, symbol: 'circle', symbolSize: 7,
        lineStyle: { color: accent, width: 2.5 }, itemStyle: { color: accent },
        label: { show: true, position: 'top', color: ink, fontFamily: 'JBMono', fontSize: 11, formatter: '{c}' } }
    ]
  }));

  // ---------- Mermaid ----------
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });
  }
})();
