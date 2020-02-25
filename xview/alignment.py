import numpy as np
import cv2


def align_post_image(pre, post):
    warpMatrix = np.zeros((3, 3), dtype=np.float32)
    warpMatrix[0, 0] = warpMatrix[1, 1] = warpMatrix[2, 2] = 1.0

    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)

    retval = False
    post_warped: np.ndarray = None

    try:
        retval, warpMatrix = cv2.findTransformECC(
            cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(post, cv2.COLOR_RGB2GRAY),
            warpMatrix,
            cv2.MOTION_HOMOGRAPHY,
            stop_criteria,
            None,
            5,
        )
        post_warped = cv2.warpPerspective(post, warpMatrix, dsize=(1024, 1024), flags=cv2.WARP_INVERSE_MAP)
    except:
        retval = False
        post_warped = post.copy()

    return post_warped




def align_post_image_pyramid(pre, post):
    pre_pyrs = [cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY)]
    pre_pyrs.append(cv2.pyrDown(pre_pyrs[-1]))
    pre_pyrs.append(cv2.pyrDown(pre_pyrs[-1]))
    pre_pyrs.append(cv2.pyrDown(pre_pyrs[-1]))

    post_pyrs = [cv2.cvtColor(post, cv2.COLOR_RGB2GRAY)]
    post_pyrs.append(cv2.pyrDown(post_pyrs[-1]))
    post_pyrs.append(cv2.pyrDown(post_pyrs[-1]))
    post_pyrs.append(cv2.pyrDown(post_pyrs[-1]))

    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    warpMatrix = np.zeros((3, 3), dtype=np.float32)
    warpMatrix[0, 0] = warpMatrix[1, 1] = warpMatrix[2, 2] = 1.0

    scale_up = np.zeros((3, 3), dtype=np.float32)
    scale_up[0, 0] = scale_up[1, 1] = 0.5
    scale_up[2, 2] = 1.0

    M = np.zeros((3, 3), dtype=np.float32)
    M[0, 0] = M[1, 1] = M[2, 2] = 1.0

    for pre_i, post_i in zip(reversed(pre_pyrs), reversed(post_pyrs)):
        warpMatrix = np.zeros((3, 3), dtype=np.float32)
        warpMatrix[0, 0] = warpMatrix[1, 1] = warpMatrix[2, 2] = 1.0

        retval = False

        post_i_refined = cv2.warpPerspective(post_i, M,
              dsize=(post_i.shape[1], post_i.shape[0]),
              flags=cv2.WARP_INVERSE_MAP)

        try:
            retval, warpMatrix = cv2.findTransformECC(
                pre_i,
                post_i_refined,
                warpMatrix,
                cv2.MOTION_HOMOGRAPHY,
                stop_criteria,
                None,
                5,
            )

            if retval:
                M = np.dot(warpMatrix, M)
                # M = np.dot(np.dot(scale_up, warpMatrix), M)
                # M = np.dot(np.dot(warpMatrix, scale_up), M)
                # M = np.dot(M, np.dot(warpMatrix, scale_up))
                # M = np.dot(M, np.dot(scale_up, warpMatrix))
        except:
            pass

    post_warped = cv2.warpPerspective(post, M,
                                         dsize=(post.shape[1], post.shape[0]),
                                         flags=cv2.WARP_INVERSE_MAP)

    return post_warped
