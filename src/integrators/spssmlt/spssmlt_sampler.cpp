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

#include "spssmlt_sampler.h"

MTS_NAMESPACE_BEGIN

SPSSMLTSampler::SPSSMLTSampler(const SPSSMLTConfiguration &config) : Sampler(Properties()) {
  m_random = new Random();
  m_s1 = config.mutationSizeLow;
  m_s2 = config.mutationSizeHigh;
  configure();
}

SPSSMLTSampler::SPSSMLTSampler(SPSSMLTSampler *sampler) : Sampler(Properties()),
                                                          m_random(sampler->m_random) {
  m_s1 = sampler->m_s1;
  m_s2 = sampler->m_s2;
  configure();
}

SPSSMLTSampler::SPSSMLTSampler(Stream *stream, InstanceManager *manager)
        : Sampler(stream, manager) {
  m_random = static_cast<Random *>(manager->getInstance(stream));
  m_s1 = stream->readFloat();
  m_s2 = stream->readFloat();

  SLog(EError, "The SPSSMLT sampler is not serializable yet");

  configure();
}

void SPSSMLTSampler::serialize(Stream *stream, InstanceManager *manager) const {
  Sampler::serialize(stream, manager);
  manager->serialize(stream, m_random.get());
  stream->writeFloat(m_s1);
  stream->writeFloat(m_s2);

  SLog(EError, "The SPSSMLT sampler is not serializable yet");
}

void SPSSMLTSampler::configure() {
  m_logRatio = -math::fastlog(m_s2 / m_s1);
  m_time = 0;
  m_largeStepTime = 0;
  m_largeStep = false;
  m_sampleIndex = 0;
  m_sampleCount = 0;
  replay = false;
  amcmc = new AMCMC;
}

SPSSMLTSampler::~SPSSMLTSampler() {}

void SPSSMLTSampler::accept() {
  if (m_largeStep)
    m_largeStepTime = m_time;
  m_time++;

  m_backup.clear();
  m_sampleIndex = 0;

  if (!m_largeStep && !replay) {
    amcmc->nbMut++;
    amcmc->nbAccMut++;
    updateMutationScale();
  }
}

void SPSSMLTSampler::reset() {
  m_time = m_sampleIndex = m_largeStepTime = 0;
  m_u.clear();
}

void SPSSMLTSampler::reject() {
  for (size_t i = 0; i < m_backup.size(); ++i)
    m_u[m_backup[i].first] = m_backup[i].second;
  m_backup.clear();
  m_sampleIndex = 0;

  if (!m_largeStep && !replay) {
    amcmc->nbMut++;
    updateMutationScale();
  }
}

Float SPSSMLTSampler::primarySample(size_t i) {
  while (i >= m_u.size()) {
    m_u.push_back(SampleStruct(m_random->nextFloat()));
  }

  if (m_u[i].modify < m_time) {
    if (i == 0 && replay) {
      SLog(EError, "Impossible");
    }

    if (m_largeStep || force_uniform) {
      m_backup.push_back(std::pair<size_t, SampleStruct>(i, m_u[i]));
      m_u[i].modify = m_time;
      m_u[i].value = m_random->nextFloat();
    } else {
      if (m_u[i].modify < m_largeStepTime) {
        m_u[i].modify = m_largeStepTime;
        m_u[i].value = m_random->nextFloat();
      }

      while (m_u[i].modify + 1 < m_time) {
        m_u[i].value = mutate(m_u[i].value);
        m_u[i].modify++;
      }

      m_backup.push_back(std::pair<size_t, SampleStruct>(i, m_u[i]));

      m_u[i].value = mutate(m_u[i].value);
      m_u[i].modify++;
    }
  }

  return m_u[i].value;
}

ref<Sampler> SPSSMLTSampler::clone() {
  ref<SPSSMLTSampler> sampler = new SPSSMLTSampler(this);
  sampler->m_sampleCount = m_sampleCount;
  sampler->m_sampleIndex = m_sampleIndex;
  sampler->m_random = new Random(m_random);

  return sampler.get();
}

ref<Sampler> SPSSMLTSampler::copy() {
  ref<SPSSMLTSampler> sampler = new SPSSMLTSampler(this);
  sampler->m_sampleCount = m_sampleCount;
  sampler->m_sampleIndex = m_sampleIndex;
  sampler->m_random = new Random(m_random);

  sampler->m_s1 = m_s1;
  sampler->m_s2 = m_s2;
  sampler->m_logRatio = m_logRatio;
  sampler->m_largeStep = m_largeStep;
  sampler->m_u = m_u;
  sampler->m_time = m_time;
  sampler->m_largeStepTime = m_largeStepTime;
  sampler->replay = replay;

  return sampler.get();
}

Float SPSSMLTSampler::next1D() {
  Float value1 = primarySample(m_sampleIndex++);
  return value1;
}

Point2 SPSSMLTSampler::next2D() {
  /// Enforce a specific order of evaluation
  Float value1 = primarySample(m_sampleIndex++);
  Float value2 = primarySample(m_sampleIndex++);

  return Point2(value1, value2);
}

std::string SPSSMLTSampler::toString() const {
  std::ostringstream oss;
  oss << "PSSMLTSampler[" << endl
      << "  sampleCount = " << m_sampleCount << endl
      << "]";
  return oss.str();
}

MTS_IMPLEMENT_CLASS_S(SPSSMLTSampler, false, Sampler)
MTS_NAMESPACE_END
