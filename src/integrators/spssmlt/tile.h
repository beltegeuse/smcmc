#ifndef MITSUBA_SPPSMLT_ABSTRACTTILE_H
#define MITSUBA_SPPSMLT_ABSTRACTTILE_H

#include "spssmlt.h"
#include "spssmlt_sampler.h"

#include <vector>
#include <limits>

MTS_NAMESPACE_BEGIN

class AbstractTile {
    /************************************************************************
     * Internal Structs
     ************************************************************************/
public:

    enum ETileValue {
        ECur = 0,
        EUp = 1,
        ERight = 2,
        EDown = 3,
        ELeft = 4
    };

    struct Pixel {
        Point2i pos;
        Spectrum values = Spectrum(0.f);
        Spectrum values_MC = Spectrum(0.f);

        void serialize(Stream *stream) const {
            pos.serialize(stream);
            values.serialize(stream);
            values_MC.serialize(stream);
        };

        void deserialize(Stream *stream) {
            pos = Point2i(stream);
            values = Spectrum(stream);
            values_MC = Spectrum(stream);
        };
    };


    /************************************************************************
     * Instance Variables
     ************************************************************************/
public:
    // Tile informations
    std::vector<Pixel> pixels;
    int nbSamples; // Total amount of samples

    // Local normalization of the tilde
    Float normalization;
    int nbSamplesUni;

    // The scale of the current tilde
    Float scale;

    // Store the state
    int groupID;

    int REAcc = 0;
    int REAttempt = 0;

    /************************************************************************
     * Public Constructors
     ************************************************************************/
public:

    AbstractTile(const Point2i &imgPos, SPSSMLTSampler *_sampler) :
            sampler(_sampler), pos(imgPos), pixels(5, Pixel()) {

        // Pixels statistics
        pixels[ECur].pos = Point2i(imgPos);
        pixels[EUp].pos = Point2i(imgPos + Point2i(0, -1));
        pixels[ERight].pos = Point2i(imgPos + Point2i(1, 0));
        pixels[EDown].pos = Point2i(imgPos + Point2i(0, 1));
        pixels[ELeft].pos = Point2i(imgPos + Point2i(-1, 0));

        // The rest of statistics
        scale = 1.f;
        normalization = 0.f;
        nbSamplesUni = 0;
        nbSamples = 0;
        groupID = 0;

        // Initialization statistics
        cumulativeWeight = 0.0;
        impCurr = 0.f;
        impProp = 0.f;
        current = std::vector<Spectrum>(5, Spectrum(0.f));
        proposed = std::vector<Spectrum>(5, Spectrum(0.f));
    }

    virtual int getNormSamples() const {
        return nbSamplesUni;
    }

    virtual Float getNorm() const {
        if (nbSamplesUni == 0.f) {
            return 0.f;
        }
        return normalization / nbSamplesUni;
    }

    // Accepting or rejecting a state
    void accept(Float proposedWeight) {
        // Accumulate the results inside the tilde
        if (cumulativeWeight != 0 && impCurr != 0.0)
            this->accum(current, cumulativeWeight / impCurr);

        // Change the current state
        cumulativeWeight = proposedWeight;
        impCurr = impProp;
        impProp = 0;
        for (int i = 0; i < this->size(); i++) {
            current[i] = proposed[i];
            proposed[i] = Spectrum(0.f);
        }

        // Accept the move and compute some statistics
        sampler->accept();
    }

    void reject(Float proposedWeight) {
        // Accumulate the results inside the tilde
        if (proposedWeight != 0 && impProp != 0.0)
            this->accum(proposed, proposedWeight / impProp);
        sampler->reject();
    }

    void flush() {
        if (impCurr != 0.f && cumulativeWeight != 0.0) {
            accum(current, cumulativeWeight / impCurr);
            cumulativeWeight = 0.f;
        }
    }

    void initializeDeadPixels() {
        for (int i = 0; i < size(); i++) {
            if (pixels[i].values.isZero())
                pixels[i].values = Spectrum(1e-10);
        }
    }

public:
    SPSSMLTSampler *sampler;

    // The proposed and current state
    Float cumulativeWeight;
    std::vector<Spectrum> current;
    Float impCurr;
    std::vector<Spectrum> proposed;
    Float impProp;
    Point2i pos;

    // Statistics RE / Small move
    int nbSmallMut = 0;
    int nbSmallMutAcc = 0;
    bool REInit = false;
    bool OriInit = false;


    /************************************************************************
     * Data Functions
     ************************************************************************/
public:

    //pixels[i].values * scale
    Spectrum get(int i) const {
        if (i < 0 && i >= size()) {
            SLog(EError, "Size error");
        }
        return pixels[i].values * scale;
    }

    //Luminance of the ith pixel
    Float lum(int i) const {
        return get(i).getLuminance();
    }

    //Average luminance of the entire
    virtual Float lum() const {
        Float lumSum = 0.0;

        for (Pixel p : pixels)
            lumSum += (p.values.getLuminance() * scale);

        lumSum /= pixels.size();

        return lumSum;
    }

    //Pixel count
    size_t size() const {
        return pixels.size();
    }

    void applyScale(Float v) {
        if (!std::isfinite(v)) {
            SLog(EWarn, "Scale infine problem");
        } else {
            scale *= v;
        }
    }

    void applyScaleColor(Float v, int channel) {
        for (size_t i = 0; i < size(); i++) {
            pixels[i].values[channel] *= v;
        }
    }


    virtual void accumNorm(Float imp, const std::vector<Spectrum>& v) {
        normalization += imp;
        nbSamplesUni += 1;
        for (size_t i = 0; i < size(); i++) {
            pixels[i].values_MC += v[i];
        }
    }

    void newSample() {
        nbSamples += 1;
    }

    void scaleNbSamples() {
        if (nbSamples != 0) {
            applyScale(1.0 / (Float) nbSamples);
        } else {
            applyScale(1.0);
        }
    }

    void resetScale() {
        scale = 1.f;
    }

    Float getScale() const {
        return scale;
    }

    //Returns the 2D point of a pixel via its internal index
    Point2i pixel(int i) const {
        SAssert(i >= 0 && i < size());
        switch (i) {
            case 0:
                return pos;
            case 1:
                return Point2i(pos.x, pos.y - 1);
            case 2:
                return Point2i(pos.x + 1, pos.y);
            case 3:
                return Point2i(pos.x, pos.y + 1);
            case 4:
                return Point2i(pos.x - 1, pos.y);
            default:
                SLog(EError, "Bad position");
        }

    }

    void accum(const std::vector<Spectrum> &res, Float factor) {
        SAssert(res.size() == size());
        for (size_t i = 0; i < size(); i++) {
            Spectrum sample = res[i] * factor;
            pixels[i].values += sample;
        }
    }


    void serialize(Stream *stream) {
        stream->writeLong(pixels.size());
        for (Pixel &p : pixels)
            p.serialize(stream);

        stream->writeFloat(normalization);
        stream->writeInt(nbSamples);
        stream->writeInt(nbSamplesUni);
        stream->writeFloat(scale);
        stream->writeInt(groupID);
        stream->writeFloat(cumulativeWeight);
        stream->writeFloat(impCurr);
        stream->writeFloat(impProp);
        pos.serialize(stream);

        stream->writeLong(current.size());
        for (Spectrum &s : current)
            s.serialize(stream);

        stream->writeLong(proposed.size());
        for (Spectrum &s : proposed)
            s.serialize(stream);
    }

    void deserialize(Stream *stream) {
        long pixelsLen = stream->readLong();
        pixels.clear();
        for (int i = 0; i < pixelsLen; ++i) {
            Pixel p;
            p.deserialize(stream);
            pixels.push_back(p);
        }

        normalization = stream->readFloat();
        nbSamples = stream->readInt();
        nbSamplesUni = stream->readInt();
        scale = stream->readFloat();
        groupID = stream->readInt();
        cumulativeWeight = stream->readFloat();
        impCurr = stream->readFloat();
        impProp = stream->readFloat();
        pos = Point2i(stream);

        long currLen = stream->readLong();
        current.clear();
        for (int i = 0; i < currLen; ++i)
            current.push_back(Spectrum(stream));

        long propLen = stream->readLong();
        proposed.clear();
        for (int i = 0; i < propLen; ++i)
            proposed.push_back(Spectrum(stream));
    }
};

// Helper function for getting the tildes for a given op
inline int getIMpos(const Vector2i &imgSize, const Point2i &pos) {
    if (pos.x < 0 || pos.x >= imgSize.x || pos.y < 0 || pos.y >= imgSize.y)
        return -1; // Invalid ID

    return pos.y * imgSize.x + pos.x;
}

inline AbstractTile &getTilde(std::vector<AbstractTile *> &tildes, const Vector2i &imgSize, const Point2i &pos) {
    return *tildes[getIMpos(imgSize, pos)];
}

MTS_NAMESPACE_END

#endif //MITSUBA_SPPSMLT_ABSTRACTTILE_H
