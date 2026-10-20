// 增补篇图表初始化
(function () {
  var s = getComputedStyle(document.documentElement);
  var accent = s.getPropertyValue('--accent').trim();
  var accent2 = s.getPropertyValue('--accent2').trim();
  var ok = s.getPropertyValue('--ok').trim();
  var ink = s.getPropertyValue('--ink').trim();
  var muted = s.getPropertyValue('--muted').trim();
  var rule = s.getPropertyValue('--rule').trim();
  var bg2 = s.getPropertyValue('--bg2').trim();

  if (typeof echarts === 'undefined') return;
  var el = document.getElementById('chart-paths');
  if (!el) return;

  var chart = echarts.init(el, null, { renderer: 'svg' });
  chart.setOption({
    animation: false,
    textStyle: { fontFamily: 'Instrument, sans-serif', color: ink },
    tooltip: {
      trigger: 'item', appendToBody: true, backgroundColor: bg2, borderColor: rule,
      textStyle: { color: ink, fontSize: 12 },
      formatter: function (p) {
        var notes = {
          'MoE + 专家卸载': 'Strata 已在 12GB 跑 125B MoE；WiSP 卸载吞吐 2.0×',
          '激活稀疏 + CUDA 核': 'TwELL 推理 +20.5%；2:4 激活稀疏可无损加速',
          '投机解码 / 提前退出': 'Cassandra（边缘自投机）、FlexEE（卸载感知早退）',
          '事件驱动稀疏硬件': 'Loihi 2 果蝇全脑仿真；通用化仍属开放问题',
          '皮层柱级类脑建模': 'MICrONS 提供实证基础，工程化未知'
        };
        return '<b>' + p.name + '</b><br/>成熟度 ' + p.value + ' / 5<br/>' + (notes[p.name] || '');
      }
    },
    grid: { left: 12, right: 60, top: 20, bottom: 20, containLabel: true },
    xAxis: {
      type: 'value', min: 0, max: 5, splitLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted }, name: '成熟度', nameTextStyle: { color: muted }
    },
    yAxis: {
      type: 'category',
      data: ['皮层柱级类脑建模', '事件驱动稀疏硬件', '投机解码 / 提前退出', '激活稀疏 + CUDA 核', 'MoE + 专家卸载'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: ink, fontSize: 12 }
    },
    series: [{
      type: 'bar', barWidth: '52%',
      data: [
        { value: 1, itemStyle: { color: muted + '66' } },
        { value: 2, itemStyle: { color: accent2 + 'aa' } },
        { value: 4, itemStyle: { color: ok } },
        { value: 4, itemStyle: { color: accent } },
        { value: 4.5, itemStyle: { color: accent } }
      ],
      label: { show: true, position: 'right', color: ink, fontFamily: 'JBMono', fontSize: 12, formatter: '{c}' }
    }]
  });
  window.addEventListener('resize', function () { chart.resize(); });
})();
