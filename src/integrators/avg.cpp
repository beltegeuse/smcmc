// STL includes
#include <fstream>

// MTS includes
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>

#include <random>

MTS_NAMESPACE_BEGIN

class AvgIntegrator : public Integrator {
public:
  AvgIntegrator(const Properties &props) : Integrator(props) {
    m_maxPass = props.getInteger("maxPass", -1);
    m_maxRenderingTime = props.getInteger("maxRenderingTime", -1);
    m_randomSampler = props.getBoolean("randomSampler", false);
  }

  bool preprocess(const Scene *scene, RenderQueue *queue,
      const RenderJob *job, int sceneResID, int sensorResID,
      int samplerResID) {
    m_subIntegrator->preprocess(scene, queue, job, sceneResID, sensorResID, samplerResID);
    m_stop = false;
    return true;
  }

  bool hitMaxPass(int currentIt) const {
    if(m_maxPass == -1) return false;
    else return currentIt >= m_maxPass;
  }
  bool hitMaxTime(Float cumlTime) {
    if(m_maxRenderingTime == -1) return false;
    else return cumlTime >= m_maxRenderingTime;
  }

  bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
      int sceneResID, int sensorResID, int samplerResID) {
    /// Get data
    ref<Scheduler> sched = Scheduler::getInstance();
    ref<Sensor> sensor = scene->getSensor();
    ref<Film> film = sensor->getFilm();
    size_t nCores = sched->getCoreCount();
    Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..",
      film->getCropSize().x, film->getCropSize().y,
      nCores, nCores == 1 ? "core" : "cores");

    Vector2i cropSize = film->getCropSize();

    if(m_randomSampler) {
      auto props = Properties("independent");
      props.setInteger("sampleCount", 256);
      ref<Sampler> indepSampler = static_cast<Sampler *> (PluginManager::getInstance()->
              createObject(MTS_CLASS(Sampler), props));
      std::random_device rd;
      std::mt19937_64 gen(rd());

      /* This is where you define the number generator for unsigned long long: */
      std::uniform_int_distribution<unsigned long long> dis;

      ref<Random> random = new Random();
      random->incRef();
      random->seed(dis(gen));
      indepSampler->setRandomSampler(random);
      sched->unregisterResource(samplerResID);

      indepSampler->incRef();
      samplerResID = sched->registerResource(indepSampler);
    }

    /// Create bitmaps
    ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
    ref<Bitmap> bitmapAvg = bitmap->clone();
    bitmapAvg->clear();

    /// Create all struct
    std::string timeFilename = scene->getDestinationFile().string()
            + "_time.csv";
    std::ofstream timeFile(timeFilename.c_str());
    ref<Timer> renderingTimer = new Timer;
    Float cumulativeTime = 0.f;

    for(int it = 0; (!hitMaxPass(it)) && (!m_stop) && (!hitMaxTime(cumulativeTime)); it++) {
      bitmap->clear(); //< Clear bitmap this pass
      film->clear(); //< and setup

      // TODO: Do something with samplers ?

      /// Render
      m_subIntegrator->render(scene, queue, job, sceneResID, sensorResID, samplerResID);
      film->develop(Point2i(0,0), cropSize, Point2i(0,0) ,bitmap);

      /// Avg results
      if(!m_stop) {
        bitmapAvg->scale(it);
        bitmapAvg->accumulate(bitmap);
        bitmapAvg->scale(1.f/(it+1.f));

        /// Write image
        {
          /// Path computation
          std::stringstream ss;
          ss << scene->getDestinationFile().c_str() << "_pass_" << (it+1);
          std::string path = ss.str();

          /// Develop image
          film->setBitmap(bitmapAvg);
          film->setDestinationFile(path,0);
          film->develop(scene,0.f);

          /// Revert destination file
          film->setDestinationFile(scene->getDestinationFile(),0);
        }

        /// Time it
        unsigned int milliseconds = renderingTimer->getMilliseconds();
        timeFile << (milliseconds / 1000.f) << ",\n";
        timeFile.flush();
        Log(EInfo, "Rendering time: %i, %i", milliseconds / 1000,
          milliseconds % 1000);
        cumulativeTime += (milliseconds / 1000.f);
        renderingTimer->reset();

        // === Print the statistic at each step of the rendering
        // to see the algorithm behaviors.
        Statistics::getInstance()->printStats();
      }
    }

    return true;
  }

  void cancel() {
    m_subIntegrator->cancel();
    m_stop = true;
  }

  ///////////////
  // Config subintergrator
  ///////////////
  void configureSampler(const Scene *scene, Sampler *sampler) {
    m_subIntegrator->configureSampler(scene, sampler);
  }

  void bindUsedResources(ParallelProcess *proc) const {
    m_subIntegrator->bindUsedResources(proc);
  }

  void wakeup(ConfigurableObject *parent,
      std::map<std::string, SerializableObject *> &params) {
    m_subIntegrator->wakeup(this, params);
  }

  void addChild(const std::string &name, ConfigurableObject *child) {
    const Class *cClass = child->getClass();

    if (cClass->derivesFrom(MTS_CLASS(Integrator))) {
      m_subIntegrator = static_cast<Integrator *>(child);
      m_subIntegrator->setParent(this);
    } else {
      Integrator::addChild(name, child);
    }
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "AvgIntegrator[" << endl
      << "  subIntegrator = " << indent(m_subIntegrator->toString()) << "," << endl
      << "]";
    return oss.str();
  }

  MTS_DECLARE_CLASS()

protected:
  int m_maxPass;
  float m_maxRenderingTime;
  ref<Integrator> m_subIntegrator;
  bool m_stop;
  bool m_randomSampler;
};

MTS_IMPLEMENT_CLASS(AvgIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(AvgIntegrator, "Avg pass integrator");
MTS_NAMESPACE_END
