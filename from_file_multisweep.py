from nuscenes.utils.geometry_utils import view_points, transform_matrix



#该函数可以添加到nuscenes的radarpointcloud中，原devkit中只有加载单个sweep函数
#该函数可以加载多个sweep，先做时间上的补偿。

@classmethod
def from_file_multisweep(cls,
                         nusc: 'NuScenes',
                         sample_rec: Dict,
                         chan: str,
                         ref_chan: str,
                         nsweeps: int = 26,
                         min_distance: float = 1.0,
                         merge: bool = True):
    """
    Return a point cloud that aggregates multiple sweeps.
    As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
    As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
    :param nusc: A NuScenes instance.
    :param sample_rec: The current sample.
    :param chan: The radar channel from which we track back n sweeps to aggregate the point cloud.
    :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
    :param nsweeps: Number of sweeps to aggregated.
    :param min_distance: Distance below which points are discarded.
    :param merge: Merge point clouds to one point cloud
    :return: (all_pc, all_times). The aggregated point cloud and timestamps.
    """

    # Init
    points = np.zeros((cls.nbr_dims(), 0))
    all_pc = cls(points)
    all_times = np.zeros((1, 0))

    # Get reference pose and timestamp
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transform from ego car frame to reference frame
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

    # Homogeneous transformation matrix from global to _current_ ego car frame
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data'][chan]
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    sweep_pcs = []
    for _ in range(nsweeps):
        # Load up the pointcloud.
        current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Remove close points and add timevector.
        current_pc.remove_close(min_distance)
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # positive difference
        times = time_lag * np.ones((1, current_pc.nbr_points()))
        all_times = np.hstack((all_times, times))

        sweep_pcs.insert(0,current_pc)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    if merge:
        # Merge with key pc.
        for sweep_pc in sweep_pcs:
            all_pc.points = np.hstack((all_pc.points, sweep_pc.points))
        
        all_pc = [all_pc]
    else:
        # we are good to go
        all_pc = sweep_pcs

    return all_pc, all_times