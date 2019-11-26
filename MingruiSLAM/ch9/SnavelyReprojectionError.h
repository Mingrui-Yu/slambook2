#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"


class SnavelyReprojectionError{
public:
    SnavelyReprojectionError(double observation_x, double observation_y):
        observed_x(observation_x), observed_y(observation_y) {}

    template<typename T>
    bool operator()(const T *const camera, const T *const point, T *residuals) const{
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }


    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions){
        T p[3];
        AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];

        const T &l1 = camera[7];
        const T &l2 = camera[8];
        const T &focal = camera[6];

        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2*(l1 + l2*r2);

        predictions[0] = focal * distortion * xp;
        predictions[1] = focal * distortion * yp;

        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y){
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x, observed_y;
};




#endif // SnavelyReprojection.h