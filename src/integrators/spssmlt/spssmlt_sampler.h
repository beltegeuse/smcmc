/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#if !defined(__PSSMLT_SAMPLER_H)
#define __PSSMLT_SAMPLER_H

#include <mitsuba/render/sampler.h>
#include <mitsuba/core/random.h>
#include "spssmlt.h"

MTS_NAMESPACE_BEGIN

struct AMCMC {
    size_t nbMut = 0;
    size_t nbAccMut = 0;
    Float scale = 1.0f;
};

/**
 * Sampler implementation as described in
 * 'A Simple and Robust Mutation Strategy for the
 * Metropolis Light Transport Algorithm' by Kelemen et al.
 */
class SPSSMLTSampler : public Sampler {
public:
    // Construct a new MLT sampler
    SPSSMLTSampler(const SPSSMLTConfiguration &conf);

    /**
     * \brief Construct a new sampler, which operates on the
     * same random number generator as \a sampler.
     */
    SPSSMLTSampler(SPSSMLTSampler *sampler);

    /// Unserialize from a binary data stream
    SPSSMLTSampler(Stream *stream, InstanceManager *manager);

    /// Set up the internal state
    void configure();

    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const;

    /// Set whether the current step should be large
    inline void setLargeStep(bool value) { m_largeStep = value; }

    /// Check if the current step is a large step
    inline bool isLargeStep() const { return m_largeStep; }

    // Make possible to replay the sequence without
    // regenerating the samples values
    inline void setupRelaySeq() {
      m_sampleIndex = 0;
    }

    inline void replayPrevious() {
      if (m_time <= 0) {
        SLog(EError, "Cannot play previous");
      } else {
        m_time--;
      }
      m_sampleIndex = 0;
      replay = true;
    }

    inline void fixTime() {
      m_time++;
    }

    inline void endReplay() {
      m_sampleIndex = 0; // Otherwise wrong
      replay = false;
    }

    /// Retrieve the next component value from the current sample
    virtual Float next1D();

    /// Retrieve the next two component values from the current sample
    virtual Point2 next2D();

    /// Return a string description
    virtual std::string toString() const;

    /// 1D mutation routine
    inline Float mutate(Float value) {
      if (HachisukaMut) {
        Float sample = m_random->nextFloat();
        bool add;

        if (sample < 0.5f) {
          add = true;
          sample *= 2.0f;
        } else {
          add = false;
          sample = 2.0f * (sample - 0.5f);
        }
        Float dv = powf(sample, (1.0 / (amcmc->scale)) + 1);


        if (add) {
          value += dv;
          if (value > 1)
            value -= 1;
        } else {
          value -= dv;
          if (value < 0)
            value += 1;
        }

        return value;
      } else {
        // Using kelemen formulation
        Float sample = m_random->nextFloat();
        bool add;

        if (sample < 0.5f) {
          add = true;
          sample *= 2.0f;
        } else {
          add = false;
          sample = 2.0f * (sample - 0.5f);
        }

        Float dv = m_s2 * math::fastexp(sample * m_logRatio);

        // This is just wrap around ...
        if (add) {
          value += dv;
          if (value > 1)
            value -= 1;
        } else {
          value -= dv;
          if (value < 0)
            value += 1;
        }

        return value;
      }
    }

    /// Return a primary sample
    Float primarySample(size_t i);

    /// Reset (& start with a large mutation)
    void reset();

    /// Accept a mutation
    void accept();

    /// Reject a mutation
    void reject();

    /// Replace the underlying random number generator
    inline void setRandom(Random *random) { m_random = random; }

    inline bool isUnitialized() const {
      return m_u.size() == 0;
    }

    /// Return the underlying random number generator
    inline Random *getRandom() { return m_random; }

    /* The following functions do nothing in this implementation */
    virtual void advance() {}

    virtual void generate(const Point2i &pos) {}

    /* The following functions are unsupported by this implementation */
    void request1DArray(size_t size) { Log(EError, "request1DArray(): Unsupported!"); }

    void request2DArray(size_t size) { Log(EError, "request2DArray(): Unsupported!"); }

    void setSampleIndex(size_t sampleIndex) { Log(EError, "setSampleIndex(): Unsupported!"); }

    ref<Sampler> clone();

    ref<Sampler> copy();

    void copy_from(SPSSMLTSampler *o) {
      m_u.clear();
      for (size_t t = 0; t < o->m_u.size(); t++) {
        m_u.push_back(o->m_u[t]);
      }
      m_time = o->m_time;
      m_largeStepTime = o->m_largeStepTime;
      m_sampleIndex = 0; // Prepare to 0
    }

    void updateMutationScale() {
      Float currAcc = amcmc->nbAccMut / (Float) amcmc->nbMut;
      amcmc->scale = amcmc->scale + ((currAcc - aStar) / sqrt(amcmc->nbMut));
      amcmc->scale = std::max(amcmc->scale, 0.0001);
    }

    void reinitAMCMC() {
      amcmc->scale = 1.f;
      amcmc->nbMut = 0;
      amcmc->nbAccMut = 0;
    }

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~SPSSMLTSampler();

protected:
    struct SampleStruct {
        Float value;
        size_t modify;

        inline SampleStruct(Float value) : value(value), modify(0) {}
    };

    ref<Random> m_random;
    Float m_s1, m_s2, m_logRatio;
    bool m_largeStep;
    std::vector<std::pair<size_t, SampleStruct> > m_backup;
    std::vector<SampleStruct> m_u;
    size_t m_time, m_largeStepTime;
    bool replay = false; // DEBUG flag

public:
    bool HachisukaMut = true;
    Float aStar = 0.234;
    bool force_uniform = false;

    AMCMC *amcmc = nullptr;

};

MTS_NAMESPACE_END

#endif /* __PSSMLT_SAMPLER_H */
