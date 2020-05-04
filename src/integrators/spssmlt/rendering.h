#ifndef MITSUBA_SPSSMLT_RENDERING_H
#define MITSUBA_SPSSMLT_RENDERING_H

#include <mitsuba/render/renderproc.h>
#include <mitsuba/core/bitmap.h>

#include "spssmlt.h"
#include "tile.h"
#include "spssmlt_sampler.h"

MTS_NAMESPACE_BEGIN

struct SamplingResultTile {
    int index = -1;
    Point2i coord = Point2i(-1, -1);

    SamplingResultTile(int i, Point2i p) :
            index(i), coord(p) {}
};

// The default rendering technique
class RenderingTechniqueTilde : public Object {
public:
    RenderingTechniqueTilde(const Scene *scene,
                            const SPSSMLTConfiguration &config,
                            const Vector2i &imgSize) :
            m_config(config),
            m_scene(scene),
            m_imgSize(imgSize) {}

    virtual SamplingResultTile Lpix(std::vector<Spectrum> &res,
                                    SPSSMLTSampler *sampler,
                                    const Point2i &pixel = Point2i(-1)) {
      SLog(EError, "Not implemented");
      return SamplingResultTile(-1, pixel);
    }

    MTS_DECLARE_CLASS()

    const SPSSMLTConfiguration &m_config;
    ref<const Scene> m_scene;
    Vector2i m_imgSize;
protected:
    /// Virtual destructor
    virtual ~RenderingTechniqueTilde() {}
};

static const Point2i offsets[] = {Point2i(0, 0), Point2i(0, -1), Point2i(1, 0), Point2i(0, 1), Point2i(-1, 0)};

class PathTracingTilde : public RenderingTechniqueTilde {
public:
    PathTracingTilde(const Scene *scene, const SPSSMLTConfiguration &config, const Vector2i &imgSize) :
            RenderingTechniqueTilde(scene, config, imgSize) {
    }

    SamplingResultTile Lpix(std::vector<Spectrum> &res,
                            SPSSMLTSampler *sampler,
                            const Point2i &base_pixel = Point2i(-1)) override {
        auto main_pixel = base_pixel;
        auto main_pixel_rand = sampler->next2D();
        if (main_pixel.x == -1 && main_pixel.y == -1) {
            // This case is for sampling the image coordinate with the global chain
            main_pixel = Point2i(m_imgSize.x * main_pixel_rand.x, m_imgSize.y * main_pixel_rand.y);
        }
        size_t main_pix_index = floor(sampler->next1D() * res.size());
        const int OFFSET_RAND = 3; // 2 for the pixel coordinates, 1 for the main pixel index
        const Medium* medium = m_scene->getSensor()->getMedium();

        // Do the shift mapping
        if (m_config.noShift) {
            for (int i = 0; i < res.size(); i++) {
                res[i] = Spectrum(0.0);
            }
            Point2i new_pix = main_pixel + offsets[main_pix_index];
            if(m_config.volume) {
                res[main_pix_index] = LVol(new_pix, medium, sampler, m_config.hideEmitter) * 5.f;
            } else {
                res[main_pix_index] = L(new_pix, sampler, m_config.hideEmitter) * 5.f;
            }
        } else {
            // Assume that res have the same size of the tilde
            // Call the path tracing for all the pixels
            for (int i = 0; i < res.size(); i++) {
                if(m_config.volume) {
                    res[i] = LVol(main_pixel + offsets[i], medium, sampler, m_config.hideEmitter);
                } else {
                    res[i] = L(main_pixel + offsets[i], sampler, m_config.hideEmitter);
                }
                if (i < res.size() - 1) {
                    sampler->setupRelaySeq(); // Replay the sequence
                    for (int k = 0; k < OFFSET_RAND; k++) {
                        sampler->next1D(); // Advance the random number
                    }
                }
            }
        }


        return SamplingResultTile(main_pix_index, main_pixel);
    }

    // Get the position on the image plane
    // Use the same sampler for everything
    Spectrum L(const TPoint2<int> &pos, SPSSMLTSampler *sampler, bool hideEmitter) const;

    // Functions for volume
    Spectrum LVol(const TPoint2<int> &pos, const Medium* medium,
                  SPSSMLTSampler *sampler, bool hideEmitter) const;
    void rayIntersectAndLookForEmitter(const Scene *scene, Sampler *sampler,
                                       const Medium *medium, int maxInteractions, Ray ray, Intersection &_its,
                                       DirectSamplingRecord &dRec, Spectrum &value) const;

private:
    inline double miWeight(double pdfA, double pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

};

MTS_NAMESPACE_END

#endif //MITSUBA_SPSSMLT_RENDERING_H
