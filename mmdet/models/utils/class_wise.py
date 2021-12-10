
def get_class_enlarge_num(gt_box_labels, pos_inds):
    '''
    class-wise enlarged region in levelset evolution. All weights also can be defined in 1.5x enlarged region.
    predefined here can obtation better performance.
    '''
    pos_class_label = gt_box_labels[pos_inds.nonzero()]

    ship_inds = (pos_class_label == 5).float()  # yes  = 1 or no  = 0
    storage_inds = (pos_class_label == 2).float()
    baseball_dia_inds = (pos_class_label == 11).float()
    tennis_court_inds = (pos_class_label == 8).float()
    basketball_court_inds = (pos_class_label == 13).float()
    gtf_inds = (pos_class_label == 9).float()
    bridge_inds = (pos_class_label == 12).float()
    large_vehicle_inds = (pos_class_label == 3).float()
    small_vehicle_inds = (pos_class_label == 1).float()
    helicopter_inds = (pos_class_label == 15).float()
    swimmingpool_inds = (pos_class_label == 6).float()
    roundabout_inds = (pos_class_label == 14).float()
    soceer_bf_inds = (pos_class_label == 10).float()
    plane_inds = (pos_class_label == 4).float()
    harbor_inds = (pos_class_label == 7).float()

    class_enlarge_num = plane_inds * 2.0 + baseball_dia_inds * 1.5 + bridge_inds * 1.5 + gtf_inds * 1.5 + \
                        small_vehicle_inds * 1.5 + large_vehicle_inds * 1.5 + ship_inds * 2.5 + tennis_court_inds * 2.5 + \
                        basketball_court_inds * 2.5 + storage_inds * 2.5 + soceer_bf_inds * 2.0 + roundabout_inds * 2.0 + \
                        harbor_inds * 2.5 + swimmingpool_inds * 1.5 + helicopter_inds * 1.5

    return class_enlarge_num

def get_class_levelset_weight(gt_box_labels, pos_inds):

    '''
    class-wise weight in levelset evolution for iSAID dataset,
    this weight can be predefined and adaptive tuning online.
    '''

    pos_class_label = gt_box_labels[pos_inds.nonzero()]
    ship_inds = (pos_class_label == 5).float()  # yes  = 1 or no  = 0
    storage_inds = (pos_class_label == 2).float()
    baseball_dia_inds = (pos_class_label == 11).float()
    tennis_court_inds = (pos_class_label == 8).float()
    basketball_court_inds = (pos_class_label == 13).float()
    gtf_inds = (pos_class_label == 9).float()
    bridge_inds = (pos_class_label == 12).float()
    large_vehicle_inds = (pos_class_label == 3).float()
    small_vehicle_inds = (pos_class_label == 1).float()
    helicopter_inds = (pos_class_label == 15).float()
    swimmingpool_inds = (pos_class_label == 6).float()
    roundabout_inds = (pos_class_label == 14).float()
    soceer_bf_inds = (pos_class_label == 10).float()
    plane_inds = (pos_class_label == 4).float()
    harbor_inds = (pos_class_label == 7).float()

    class_levelset_weight = plane_inds * 1.5 + baseball_dia_inds * 1.15 + bridge_inds * 1.05 + gtf_inds * 0.00005 + \
                            small_vehicle_inds * 1.1 + large_vehicle_inds * 1.15 + ship_inds * 0.000001 + tennis_court_inds * 0.000001 + \
                            basketball_court_inds * 0.000001 + storage_inds * 0.65 + soceer_bf_inds * 0.005 + roundabout_inds * 0.35 + \
                            harbor_inds * 1.25 + swimmingpool_inds * 0.00005 + helicopter_inds * 1.15

    return class_levelset_weight
