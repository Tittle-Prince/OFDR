import graphviz

def generate_final_ofdr_diagram():
    # 创建有向图，设置整体格式和高分辨率
    dot = graphviz.Digraph(comment='OFDR System Architecture', format='pdf')
    dot.attr(rankdir='LR', nodesep='0.6', ranksep='0.8', dpi='300')
    
    # 全局节点和连线样式设置
    dot.attr('node', shape='box', style='solid', fontname='Times New Roman', fontsize='12', penwidth='1.2')
    dot.attr('edge', fontname='Times New Roman', fontsize='11', penwidth='1.0', arrowsize='0.8')

    # ================= 定义节点 =================
    
    # 光源部分 (严格按照要求，只保留 NLL，无其他描述)
    dot.node('NLL', 'NLL')
    
    # 耦合器
    dot.node('C_split', 'Coupler')
    dot.node('C_5050', 'Coupler\n(50:50 Coupler)')
    
    # 环形器
    circ_style = {'shape': 'circle', 'width': '0.8', 'fixedsize': 'true'}
    dot.node('Circ1', 'Circulator 1', **circ_style)
    dot.node('Circ2', 'Circulator 2', **circ_style)
    
    # 传感阵列与参考臂 (仅保留 Reference Fiber，无光栅标记)
    dot.node('Sensing', 'Sensing Fiber Array\n||||   ||||   ...   ||||\nFBG  FBG        FBG', shape='plaintext')
    dot.node('Ref', 'Reference Fiber', shape='plaintext')
    
    # 探测与采集部分
    dot.node('PD', 'PD\n(Balanced Detector)\n(400 MHz Bandwidth)')
    dot.node('DSO', 'DSO\n(Digital Sampling\nOscilloscope)\n(5 MSa/s Sampling rate)')
    dot.node('PC', 'PC')

    # ================= 定义连线 =================
    
    # 激光器分光
    dot.edge('NLL', 'C_split')
    dot.edge('C_split', 'Circ1')
    dot.edge('C_split', 'Circ2')
    
    # 测量臂
    dot.edge('Circ1', 'Sensing', label=' 2')
    dot.edge('Sensing', 'Circ1') # 示意反射回光
    dot.edge('Circ1', 'C_5050', label=' 3')
    
    # 参考臂 (此时连接的是普通单模光纤)
    dot.edge('Circ2', 'Ref', label=' 2')
    dot.edge('Ref', 'Circ2') # 示意反射回光
    dot.edge('Circ2', 'C_5050', label=' 3')
    
    # 采集端
    dot.edge('C_5050', 'PD')
    dot.edge('PD', 'DSO')
    dot.edge('DSO', 'PC')

    # ================= 强制对齐以保证排版整洁 =================
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Circ1')
        s.node('Circ2')

    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Sensing')
        s.node('C_5050')
        s.node('Ref')

    # 渲染生成
    dot.render('ofdr_system_final', view=True)
    print("最终版架构图已生成为 ofdr_system_final.pdf")

if __name__ == '__main__':
    generate_final_ofdr_diagram()