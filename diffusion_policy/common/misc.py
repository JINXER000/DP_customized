def get_sg(hdf5_group, sg_name):
    import networkx as nx
    import json
    sg_json = hdf5_group[sg_name][()] if sg_name in hdf5_group else None
    if sg_json is None:
        return None
    sg_str = sg_json.decode('utf-8')
    sg = nx.node_link_graph(json.loads(sg_str))
    return sg

def get_biop_skill_info(sg_info):
    skill_info = None
    ## find the bimanual skill
    for skill_key in sg_info:
        if 'bimanual' in skill_key:
            skill_info = sg_info[skill_key]
            return skill_info
    raise ValueError('Bimanual skill not found in sg_info')

def get_biop_start(sg_info):
    skill_info = get_biop_skill_info(sg_info)
    pre_sg = get_sg(skill_info, 'pre_sg')
    biop_start = pre_sg.graph['idx_list'][0]
    return biop_start
