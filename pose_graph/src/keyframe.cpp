#include "keyframe.h"

// 根据 status 向量中的标志位筛选输入向量 v 的内容，并删除其中未通过筛选的元素。
template <typename Derived> // 函数模板，允许处理不同类型的 std::vector（如 vector<int>、vector<float> 等）
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++) // 遍历 v 中的每个元素
        if (status[i]) // 如果 status[i] 为真（即该元素通过筛选）
            v[j++] = v[i]; // 将 v[i] 移动到下标 j 位置，j 递增
    v.resize(j); // 缩小 v 的大小，移除未保留的元素
}

// create keyframe online
// 实现了 KeyFrame 类的构造函数，用于初始化一个关键帧对象。
// 这段代码接受多个输入参数，并将它们赋值或加工后存储到 KeyFrame 对象的成员变量中。
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           vector<double> &_point_id, int _sequence)
{
	time_stamp = _time_stamp; // 事件戳和关键帧索引
	index = _index;       
	vio_T_w_i = _vio_T_w_i;   // 保存 VIO 提供的初始位姿（平移和旋转）
	vio_R_w_i = _vio_R_w_i;   
	T_w_i = vio_T_w_i;        // 初始化帧的世界坐标位姿为 VIO 估算值
	R_w_i = vio_R_w_i;        
	origin_vio_T = vio_T_w_i; // 记录最初的 VIO 位姿，可能用于后续回溯或校正
	origin_vio_R = vio_R_w_i;  
	image = _image.clone();    // 深拷贝输入图像 _image，避免直接修改原图像
	cv::resize(image, thumbnail, cv::Size(80, 60));  //生成缩略图（thumbnail），可用于快速显示或轻量级计算
	point_3d = _point_3d;      // 将输入的 3D 点、2D 像素点、归一化点以及点 ID 直接存储为成员变量
	point_2d_uv = _point_2d_uv;         
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;          // 初始化为 false，表示当前关键帧尚未检测到回环
	loop_index = -1;           // 初始化为 -1，表示尚未与其他关键帧关联回环
	has_fast_point = false;    // 初始化为 false，表示尚未提取 FAST 特征点
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;  // 初始化为一个全零的向量，用于存储回环相关信息
	sequence = _sequence;      // 序列号，用于标识关键帧所属的序列
	computeWindowBRIEFPoint();  // 调用 computeWindowBRIEFPoint 函数，计算窗口内的 BRIEF 描述子
	computeBRIEFPoint();        // 调用 computeBRIEFPoint 函数，计算整个图像的 BRIEF 描述子
	if(!DEBUG_IMAGE)
		image.release();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
					cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
					vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE)
	{
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}

// 从已知的 2D 像素坐标（point_2d_uv）生成关键点，并使用 BRIEF 描述符提取算法计算描述符。
void KeyFrame::computeWindowBRIEFPoint()
{	
	// 创建 BriefExtractor 实例，加载 BRIEF 模式文件（BRIEF_PATTERN_FILE）。
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	// 遍历 point_2d_uv 中的点，将每个点转换为 OpenCV 的 cv::KeyPoint 对象，存储到 window_keypoints
	for(int i = 0; i < (int)point_2d_uv.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = point_2d_uv[i];
	    window_keypoints.push_back(key);
	}
	// 使用 BriefExtractor 处理图像和关键点列表，生成描述符 window_brief_descriptors
	extractor(image, window_keypoints, window_brief_descriptors);
}
// 基于特征点检测器提取关键点，并计算其 BRIEF 描述符,将特征点从像素坐标系转换到归一化坐标系。
void KeyFrame::computeBRIEFPoint()
{
	// 创建 BriefExtractor 实例
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 20; // corner detector response threshold
	if(1)
		// 默认使用 FAST 角点检测算法
		cv::FAST(image, keypoints, fast_th, true);
	else
	{
		vector<cv::Point2f> tmp_pts;
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
		for(int i = 0; i < (int)tmp_pts.size(); i++)
		{
		    cv::KeyPoint key;
		    key.pt = tmp_pts[i];
		    keypoints.push_back(key);
		}
	}
	// 计算 BRIEF 描述符，存储到 brief_descriptors
	extractor(image, keypoints, brief_descriptors);
	// 使用相机模型（m_camera->liftProjective）将像素坐标投影到归一化坐标系，存储在 keypoints_norm
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}
// 重载运算符 ()，调用 OpenCV 的 cv::xfeatures2d::BriefDescriptorExtractor 计算 BRIEF 描述符
void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}

// 在提供的一组旧关键帧特征点描述符（descriptors_old）中，
// 寻找与当前窗口描述符（window_descriptor）最相似的匹配点。
// 使用Hamming 距离作为相似度度量
// 如果找到满足条件的匹配点，返回是否找到特征点
bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
	// 计算当前窗口描述符与旧描述符之间的 Hamming 距离
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {
		// 计算两个描述符之间的 Hamming 距离
        int dis = HammingDis(window_descriptor, descriptors_old[i]);
		// 更新最佳匹配点索引（bestIndex）和最小距离（bestDist）
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    // 如果找到匹配帧，并且最小距离小于 80，则记录匹配帧坐标并返回 true
    if (bestIndex != -1 && bestDist < 80)
    {
      best_match = keypoints_old[bestIndex].pt;
      best_match_norm = keypoints_old_norm[bestIndex].pt;
      return true;
    }
    else
      return false;
}

// 对当前关键帧的所有窗口描述符（window_brief_descriptors）逐一调用 searchInAera，
// 尝试在旧关键帧中找到匹配点。保存每个窗口描述符的匹配状态（status）以及对应的匹配点坐标。


void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
								std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
	// 遍历当前帧的所有窗口描述符。
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
		// 调用 searchInAera
		// 如果找到匹配点，将状态设置为 1
		// 否则，将状态设置为 0
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
          status.push_back(1);
        else
          status.push_back(0);
		// 保存找到的匹配点的像素坐标和归一化坐标
		//（即使未找到匹配点，也会保存一个默认值 (0.f, 0.f)）。  
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }

}

// 通过 RANSAC 方法估计两帧间的基础矩阵（Fundamental Matrix）
// 使用基础矩阵剔除不满足几何约束的匹配点。
void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	// 初始化匹配点状态，默认所有点为外点。
	for (int i = 0; i < n; i++)
		status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
		// 归一化像素坐标转换为图像平面坐标
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 460.0;
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
		// 使用 OpenCV 的 findFundamentalMat 函数进行基础矩阵估计
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}
// 通过 RANSAC 方法估计相机位姿（旋转矩阵和平移向量）。
// 根据 3D 点和对应的 2D 像素点匹配对，求解 Perspective-n-Point (PnP) 问题。
void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
	//for (int i = 0; i < matched_3d.size(); i++)
	//	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
	//printf("match size %d \n", matched_3d.size());
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;

	// 相机内参与初始位姿设置：
	// 根据当前帧的 VIO 提供的位姿估计相机位姿。
	// 使用相机内外参数的变换计算初始旋转矩阵和平移向量。
    Matrix3d R_w_c = origin_vio_R * qic;
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);
	// eigen 转 opencv
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

	// 使用 OpenCV 的 solvePnPRansac 函数，基于 RANSAC 方法估计相机位姿。
    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);

    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);
	// 根据 RANSAC 的输出内点索引，更新匹配点的状态。
    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }
	// 将旋转向量 rvec 转换为旋转矩阵。
	// 使用 Eigen 库将 OpenCV 矩阵转换为 Eigen 格式。
    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);

	// 根据相机外参 qic 和 tic 修正计算得到的旋转和平移
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;

}

// 用于在当前关键帧与给定的旧关键帧（old_kf）之间寻找回环（Loop Closure）连接。
// 特征点匹配：通过 BRIEF 描述符搜索匹配点。
// 几何约束剔除：通过 PnP-RANSAC 和回环判断条件筛选合格的匹配点。
// 回环信息计算：如果满足回环条件，计算相对位姿并发布回环消息。
bool KeyFrame::findConnection(KeyFrame* old_kf)
{
	// 初始化变量
	// 初始化当前帧和旧关键帧的匹配数据存储容器。
	// 将当前帧的 3D 点、2D 点、归一化坐标以及特征点 ID 初始化为本地变量。
	TicToc tmp_t;
	//printf("find Connection\n");
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	TicToc t_match;
	#if 0
		if (DEBUG_IMAGE)
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif
	//printf("search by des\n");
	// 调用 searchByBRIEFDes 进行描述符匹配，找到当前帧的窗口描述符与旧关键帧描述符中相似的点对。
	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
	// 根据匹配状态，过滤掉不满足匹配条件的点对。
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	//printf("search by des finish\n");

	#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);
	        */

	    }
	#endif
	// 状态重置
	status.clear();
	/*
	FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	*/
	#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif
	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	// 如果匹配点数大于阈值，则进行PnP求解
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		status.clear();
		// PnP-RANSAC 几何约束筛选
	    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
	    reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	    reduceVector(matched_id, status);
	    #if 1
	    	if (DEBUG_IMAGE)
	        {
	        	int gap = 10;
	        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
	            cv::Mat gray_img, loop_match_img;
	            cv::Mat old_img = old_kf->image;
	            cv::hconcat(image, gap_image, gap_image);
	            cv::hconcat(gap_image, old_img, gray_img);
	            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	            for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f cur_pt = matched_2d_cur[i];
	                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for(int i = 0; i< (int)matched_2d_old.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap);
	                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap) ;
	                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
	            }
	            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
	            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

	            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
	            cv::vconcat(notation, loop_match_img, loop_match_img);

	            /*
	            ostringstream path;
	            path <<  "/home/tony-ws1/raw_data/loop_image/"
	                    << index << "-"
	                    << old_kf->index << "-" << "3pnp_match.jpg";
	            cv::imwrite( path.str().c_str(), loop_match_img);
	            */
	            if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	            {
	            	/*
	            	cv::imshow("loop connection",loop_match_img);
	            	cv::waitKey(10);
	            	*/
	            	cv::Mat thumbimage;
	            	cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
	    	    	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
	                msg->header.stamp = ros::Time(time_stamp);
	    	    	pub_match_img.publish(msg);
	            }
	        }
	    #endif
	}
	// 如果匹配点数大于阈值，则认为有闭环
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		// 计算当前帧和旧关键帧之间的相对位姿 
	    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old); // 相对平移
	    relative_q = PnP_R_old.transpose() * origin_vio_R;				// 相对旋转
	    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x()); // 相对偏航角
	    //printf("PNP relative\n");
	    //cout << "pnp relative_t " << relative_t.transpose() << endl;
	    //cout << "pnp relative_yaw " << relative_yaw << endl;
		// 判断回环是否满足条件：偏航角差值小于 30° 且平移距离小于 20 个单位（场景单位，如米）。
	    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
	    {
	    	// 如果满足条件，则认为有闭环，并记录旧关键帧的索引和相对位姿信息
	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
			// 如果启用了快速重定位（FAST_RELOCALIZATION），
			// 将匹配点和回环位姿信息封装为消息，并通过 ROS 发布。
	    	if(FAST_RELOCALIZATION)
	    	{
			    sensor_msgs::PointCloud msg_match_points;
			    msg_match_points.header.stamp = ros::Time(time_stamp);
			    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
			    {
		            geometry_msgs::Point32 p;
		            p.x = matched_2d_old_norm[i].x;
		            p.y = matched_2d_old_norm[i].y;
		            p.z = matched_id[i];
		            msg_match_points.points.push_back(p);
			    }
			    Eigen::Vector3d T = old_kf->T_w_i;
			    Eigen::Matrix3d R = old_kf->R_w_i;
			    Quaterniond Q(R);
			    sensor_msgs::ChannelFloat32 t_q_index;
			    t_q_index.values.push_back(T.x());
			    t_q_index.values.push_back(T.y());
			    t_q_index.values.push_back(T.z());
			    t_q_index.values.push_back(Q.w());
			    t_q_index.values.push_back(Q.x());
			    t_q_index.values.push_back(Q.y());
			    t_q_index.values.push_back(Q.z());
			    t_q_index.values.push_back(index);
			    msg_match_points.channels.push_back(t_q_index);
			    pub_match_points.publish(msg_match_points);
	    	}
	        return true;
	    }
	}
	//printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
	return false;
}

// 计算两个 BRIEF 描述符之间的 Hamming 距离，用于衡量两个描述符的相似性。 Hamming 距离越小，表示两个描述符越相似。
int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
	// 使用异或操作 ^ 对两个描述符（a 和 b）进行逐位比较，结果为 1 的位表示两个描述符在该位不同。
    BRIEF::bitset xor_of_bitset = a ^ b;
	// count() 函数计算异或结果中 1 的个数，即两个描述符不同的位数。
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		//printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}
