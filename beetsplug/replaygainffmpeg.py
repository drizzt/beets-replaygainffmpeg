# -*- coding: utf-8 -*-
# This file is part of beets.
# Copyright 2016, Fabrice Laporte, Yevgeny Bezman, and Adrian Sampson.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

from __future__ import division, absolute_import, print_function

import subprocess
import os
import collections
import math
import sys
import warnings
import enum
import platform
import re
import xml.parsers.expat
from six.moves import zip

from beets import ui, plugins
from beets.plugins import BeetsPlugin
from beets.util import (syspath, convert_command_args, bytestring_path,
                        displayable_path, py3_path)

from beetsplug.replaygain import (Backend, ReplayGainError,
                                  FatalReplayGainError,
                                  FatalGstreamerPluginReplayGainError)

# Copied from beets/util/__init__.py

# stdout and stderr as bytes
CommandOutput = collections.namedtuple("CommandOutput", ("stdout", "stderr"))


def command_output(cmd, shell=False):
    """Runs the command and returns its output after it has exited.

    Returns a CommandOutput. The attributes ``stdout`` and ``stderr`` contain
    byte strings of the respective output streams.

    ``cmd`` is a list of arguments starting with the command names. The
    arguments are bytes on Unix and strings on Windows.
    If ``shell`` is true, ``cmd`` is assumed to be a string and passed to a
    shell to execute.

    If the process exits with a non-zero return code
    ``subprocess.CalledProcessError`` is raised. May also raise
    ``OSError``.

    This replaces `subprocess.check_output` which can have problems if lots of
    output is sent to stderr.
    """
    cmd = convert_command_args(cmd)

    try:  # python >= 3.3
        devnull = subprocess.DEVNULL
    except AttributeError:
        devnull = open(os.devnull, 'r+b')

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=devnull,
        close_fds=platform.system() != 'Windows',
        shell=shell
    )
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise subprocess.CalledProcessError(
            returncode=proc.returncode,
            cmd=' '.join(cmd),
            output=stdout + stderr,
        )
    return CommandOutput(stdout, stderr)

# Copied from replaygain.py

def call(args, **kwargs):
    """Execute the command and return its output or raise a
    ReplayGainError on failure.
    """
    try:
        return command_output(args, **kwargs)
    except subprocess.CalledProcessError as e:
        raise ReplayGainError(
            u"{0} exited with status {1}".format(args[0], e.returncode)
        )
    except UnicodeEncodeError:
        # Due to a bug in Python 2's subprocess on Windows, Unicode
        # filenames can fail to encode on that platform. See:
        # https://github.com/google-code-export/beets/issues/499
        raise ReplayGainError(u"argument encoding failed")


def db_to_lufs(db):
    """Convert db to LUFS.

    According to https://wiki.hydrogenaud.io/index.php?title=
      ReplayGain_2.0_specification#Reference_level
    """
    return db - 107


def lufs_to_db(db):
    """Convert LUFS to db.

    According to https://wiki.hydrogenaud.io/index.php?title=
      ReplayGain_2.0_specification#Reference_level
    """
    return db + 107


# Backend base and plumbing classes.

# gain: in LU to reference level
# peak: part of full scale (FS is 1.0)
Gain = collections.namedtuple("Gain", "gain peak")
# album_gain: Gain object
# track_gains: list of Gain objects
AlbumGain = collections.namedtuple("AlbumGain", "album_gain track_gains")


class Peak(enum.Enum):
    none = 0
    true = 1
    sample = 2


# ffmpeg backend
class FfmpegBackend(Backend):
    """A replaygain backend using ffmpeg's ebur128 filter.
    """
    def __init__(self, config, log):
        super(FfmpegBackend, self).__init__(config, log)
        self._ffmpeg_path = "ffmpeg"

        # check that ffmpeg is installed
        try:
            ffmpeg_version_out = call([self._ffmpeg_path, "-version"])
        except OSError:
            raise FatalReplayGainError(
                u"could not find ffmpeg at {0}".format(self._ffmpeg_path)
            )
        incompatible_ffmpeg = True
        for line in ffmpeg_version_out.stdout.splitlines():
            if line.startswith(b"configuration:"):
                if b"--enable-libebur128" in line:
                    incompatible_ffmpeg = False
            if line.startswith(b"libavfilter"):
                version = line.split(b" ", 1)[1].split(b"/", 1)[0].split(b".")
                version = tuple(map(int, version))
                if version >= (6, 67, 100):
                    incompatible_ffmpeg = False
        if incompatible_ffmpeg:
            raise FatalReplayGainError(
                u"Installed FFmpeg version does not support ReplayGain."
                u"calculation. Either libavfilter version 6.67.100 or above or"
                u"the --enable-libebur128 configuration option is required."
            )

    def compute_track_gain(self, items, target_level, peak):
        """Computes the track gain of the given tracks, returns a list
        of Gain objects (the track gains).
        """
        gains = []
        for item in items:
            gains.append(
                self._analyse_item(
                    item,
                    target_level,
                    peak,
                    count_blocks=False,
                )[0]  # take only the gain, discarding number of gating blocks
            )
        return gains

    def compute_album_gain(self, items, target_level, peak):
        """Computes the album gain of the given album, returns an
        AlbumGain object.
        """
        target_level_lufs = db_to_lufs(target_level)

        # analyse tracks
        # list of track Gain objects
        track_gains = []
        # maximum peak
        album_peak = 0
        # sum of BS.1770 gating block powers
        sum_powers = 0
        # total number of BS.1770 gating blocks
        n_blocks = 0

        for item in items:
            track_gain, track_n_blocks = self._analyse_item(
                item, target_level, peak
            )
            track_gains.append(track_gain)

            # album peak is maximum track peak
            album_peak = max(album_peak, track_gain.peak)

            # prepare album_gain calculation
            # total number of blocks is sum of track blocks
            n_blocks += track_n_blocks

            # convert `LU to target_level` -> LUFS
            track_loudness = target_level_lufs - track_gain.gain
            # This reverses ITU-R BS.1770-4 p. 6 equation (5) to convert
            # from loudness to power. The result is the average gating
            # block power.
            track_power = 10**((track_loudness + 0.691) / 10)

            # Weight that average power by the number of gating blocks to
            # get the sum of all their powers. Add that to the sum of all
            # block powers in this album.
            sum_powers += track_power * track_n_blocks

        # calculate album gain
        if n_blocks > 0:
            # compare ITU-R BS.1770-4 p. 6 equation (5)
            # Album gain is the replaygain of the concatenation of all tracks.
            album_gain = -0.691 + 10 * math.log10(sum_powers / n_blocks)
        else:
            album_gain = -70
        # convert LUFS -> `LU to target_level`
        album_gain = target_level_lufs - album_gain

        self._log.debug(
            u"{0}: gain {1} LU, peak {2}"
            .format(items, album_gain, album_peak)
            )

        return AlbumGain(Gain(album_gain, album_peak), track_gains)

    def _construct_cmd(self, item, peak_method):
        """Construct the shell command to analyse items."""
        return [
            self._ffmpeg_path,
            "-nostats",
            "-hide_banner",
            "-i",
            item.path,
            "-map",
            "a:0",
            "-filter",
            "ebur128=peak={0}".format(peak_method),
            "-f",
            "null",
            "-",
        ]

    def _analyse_item(self, item, target_level, peak, count_blocks=True):
        """Analyse item. Return a pair of a Gain object and the number
        of gating blocks above the threshold.

        If `count_blocks` is False, the number of gating blocks returned
        will be 0.
        """
        target_level_lufs = db_to_lufs(target_level)
        peak_method = peak.name

        # call ffmpeg
        self._log.debug(u"analyzing {0}".format(item))
        cmd = self._construct_cmd(item, peak_method)
        self._log.debug(
            u'executing {0}', u' '.join(map(displayable_path, cmd))
        )
        output = call(cmd).stderr.splitlines()

        # parse output

        if peak == Peak.none:
            peak = 0
        else:
            line_peak = self._find_line(
                output,
                "  {0} peak:".format(peak_method.capitalize()).encode(),
                start_line=len(output) - 1, step_size=-1,
            )
            peak = self._parse_float(
                output[self._find_line(
                    output, b"    Peak:",
                    line_peak,
                )]
            )
            # convert TPFS -> part of FS
            peak = 10**(peak / 20)

        line_integrated_loudness = self._find_line(
            output, b"  Integrated loudness:",
            start_line=len(output) - 1, step_size=-1,
        )
        gain = self._parse_float(
            output[self._find_line(
                output, b"    I:",
                line_integrated_loudness,
            )]
        )
        # convert LUFS -> LU from target level
        gain = target_level_lufs - gain

        # count BS.1770 gating blocks
        n_blocks = 0
        if count_blocks:
            gating_threshold = self._parse_float(
                output[self._find_line(
                    output, b"    Threshold:",
                    start_line=line_integrated_loudness,
                )]
            )
            for line in output:
                if not line.startswith(b"[Parsed_ebur128"):
                    continue
                if line.endswith(b"Summary:"):
                    continue
                line = line.split(b"M:", 1)
                if len(line) < 2:
                    continue
                if self._parse_float(b"M: " + line[1]) >= gating_threshold:
                    n_blocks += 1
            self._log.debug(
                u"{0}: {1} blocks over {2} LUFS"
                .format(item, n_blocks, gating_threshold)
            )

        self._log.debug(
            u"{0}: gain {1} LU, peak {2}"
            .format(item, gain, peak)
        )

        return Gain(gain, peak), n_blocks

    def _find_line(self, output, search, start_line=0, step_size=1):
        """Return index of line beginning with `search`.

        Begins searching at index `start_line` in `output`.
        """
        end_index = len(output) if step_size > 0 else -1
        for i in range(start_line, end_index, step_size):
            if output[i].startswith(search):
                return i
        raise ReplayGainError(
            u"ffmpeg output: missing {0} after line {1}"
            .format(repr(search), start_line)
            )

    def _parse_float(self, line):
        """Extract a float from a key value pair in `line`.

        This format is expected: /[^:]:[[:space:]]*value.*/, where `value` is
        the float.
        """
        # extract value
        value = line.split(b":", 1)
        if len(value) < 2:
            raise ReplayGainError(
                u"ffmpeg output: expected key value pair, found {0}"
                .format(line)
                )
        value = value[1].lstrip()
        # strip unit
        value = value.split(b" ", 1)[0]
        # cast value to float
        try:
            return float(value)
        except ValueError:
            raise ReplayGainError(
                u"ffmpeg output: expected float value, found {1}"
                .format(value)
                )

class ReplayGainFfmpegPlugin(BeetsPlugin):


    backends = {
        "ffmpeg": FfmpegBackend,
    }

    peak_methods = {
        "true": Peak.true,
        "sample": Peak.sample,
    }

    def __init__(self):
        super(ReplayGainFfmpegPlugin, self).__init__()

        # default backend is 'command' for backward-compatibility.
        self.config.add({
            'overwrite': False,
            'auto': True,
            'backend': u'ffmpeg',
            'per_disc': False,
            'peak': 'true',
            'targetlevel': 89,
            'r128': ['Opus'],
            'r128_targetlevel': lufs_to_db(-23),
        })

        self.overwrite = self.config['overwrite'].get(bool)
        self.per_disc = self.config['per_disc'].get(bool)
        backend_name = self.config['backend'].as_str()
        if backend_name not in self.backends:
            raise ui.UserError(
                u"Selected ReplayGain backend {0} is not supported. "
                u"Please select one of: {1}".format(
                    backend_name,
                    u', '.join(self.backends.keys())
                )
            )
        peak_method = self.config["peak"].as_str()
        if peak_method not in self.peak_methods:
            raise ui.UserError(
                u"Selected ReplayGain peak method {0} is not supported. "
                u"Please select one of: {1}".format(
                    peak_method,
                    u', '.join(self.peak_methods.keys())
                )
            )
        self._peak_method = self.peak_methods[peak_method]

        # On-import analysis.
        if self.config['auto']:
            self.import_stages = [self.imported]

        # Formats to use R128.
        self.r128_whitelist = self.config['r128'].as_str_seq()

        try:
            self.backend_instance = self.backends[backend_name](
                self.config, self._log
            )
        except (ReplayGainError, FatalReplayGainError) as e:
            raise ui.UserError(
                u'replaygain initialization failed: {0}'.format(e))

    def should_use_r128(self, item):
        """Checks the plugin setting to decide whether the calculation
        should be done using the EBU R128 standard and use R128_ tags instead.
        """
        return item.format in self.r128_whitelist

    def track_requires_gain(self, item):
        return self.overwrite or \
            (self.should_use_r128(item) and not item.r128_track_gain) or \
            (not self.should_use_r128(item) and
                (not item.rg_track_gain or not item.rg_track_peak))

    def album_requires_gain(self, album):
        # Skip calculating gain only when *all* files don't need
        # recalculation. This way, if any file among an album's tracks
        # needs recalculation, we still get an accurate album gain
        # value.
        return self.overwrite or \
            any([self.should_use_r128(item) and
                (not item.r128_track_gain or not item.r128_album_gain)
                for item in album.items()]) or \
            any([not self.should_use_r128(item) and
                (not item.rg_album_gain or not item.rg_album_peak)
                for item in album.items()])

    def store_track_gain(self, item, track_gain):
        item.rg_track_gain = track_gain.gain
        item.rg_track_peak = track_gain.peak
        item.store()
        self._log.debug(u'applied track gain {0} LU, peak {1} of FS',
                        item.rg_track_gain, item.rg_track_peak)

    def store_album_gain(self, item, album_gain):
        item.rg_album_gain = album_gain.gain
        item.rg_album_peak = album_gain.peak
        item.store()
        self._log.debug(u'applied album gain {0} LU, peak {1} of FS',
                        item.rg_album_gain, item.rg_album_peak)

    def store_track_r128_gain(self, item, track_gain):
        item.r128_track_gain = track_gain.gain
        item.store()

        self._log.debug(u'applied r128 track gain {0} LU',
                        item.r128_track_gain)

    def store_album_r128_gain(self, item, album_gain):
        item.r128_album_gain = album_gain.gain
        item.store()
        self._log.debug(u'applied r128 album gain {0} LU',
                        item.r128_album_gain)

    def tag_specific_values(self, items):
        """Return some tag specific values.

        Returns a tuple (store_track_gain, store_album_gain, target_level,
        peak_method).
        """
        if any([self.should_use_r128(item) for item in items]):
            store_track_gain = self.store_track_r128_gain
            store_album_gain = self.store_album_r128_gain
            target_level = self.config['r128_targetlevel'].as_number()
            peak = Peak.none  # R128_* tags do not store the track/album peak
        else:
            store_track_gain = self.store_track_gain
            store_album_gain = self.store_album_gain
            target_level = self.config['targetlevel'].as_number()
            peak = self._peak_method

        return store_track_gain, store_album_gain, target_level, peak

    def handle_album(self, album, write, force=False):
        """Compute album and track replay gain store it in all of the
        album's items.

        If ``write`` is truthy then ``item.write()`` is called for each
        item. If replay gain information is already present in all
        items, nothing is done.
        """
        if not force and not self.album_requires_gain(album):
            self._log.info(u'Skipping album {0}', album)
            return

        self._log.info(u'analyzing {0}', album)

        if (any([self.should_use_r128(item) for item in album.items()]) and not
                all(([self.should_use_r128(item) for item in album.items()]))):
            self._log.error(
                u"Cannot calculate gain for album {0} (incompatible formats)",
                album)
            return

        tag_vals = self.tag_specific_values(album.items())
        store_track_gain, store_album_gain, target_level, peak = tag_vals

        discs = dict()
        if self.per_disc:
            for item in album.items():
                if discs.get(item.disc) is None:
                    discs[item.disc] = []
                discs[item.disc].append(item)
        else:
            discs[1] = album.items()

        for discnumber, items in discs.items():
            try:
                album_gain = self.backend_instance.compute_album_gain(
                    items, target_level, peak
                )
                if len(album_gain.track_gains) != len(items):
                    raise ReplayGainError(
                        u"ReplayGain backend failed "
                        u"for some tracks in album {0}".format(album)
                    )

                for item, track_gain in zip(items, album_gain.track_gains):
                    store_track_gain(item, track_gain)
                    store_album_gain(item, album_gain.album_gain)
                    if write:
                        item.try_write()
            except ReplayGainError as e:
                self._log.info(u"ReplayGain error: {0}", e)
            except FatalReplayGainError as e:
                raise ui.UserError(
                    u"Fatal replay gain error: {0}".format(e))

    def handle_track(self, item, write, force=False):
        """Compute track replay gain and store it in the item.

        If ``write`` is truthy then ``item.write()`` is called to write
        the data to disk.  If replay gain information is already present
        in the item, nothing is done.
        """
        if not force and not self.track_requires_gain(item):
            self._log.info(u'Skipping track {0}', item)
            return

        self._log.info(u'analyzing {0}', item)

        tag_vals = self.tag_specific_values([item])
        store_track_gain, store_album_gain, target_level, peak = tag_vals

        try:
            track_gains = self.backend_instance.compute_track_gain(
                [item], target_level, peak
            )
            if len(track_gains) != 1:
                raise ReplayGainError(
                    u"ReplayGain backend failed for track {0}".format(item)
                )

            store_track_gain(item, track_gains[0])
            if write:
                item.try_write()
        except ReplayGainError as e:
            self._log.info(u"ReplayGain error: {0}", e)
        except FatalReplayGainError as e:
            raise ui.UserError(
                u"Fatal replay gain error: {0}".format(e))

    def imported(self, session, task):
        """Add replay gain info to items or albums of ``task``.
        """
        if task.is_album:
            self.handle_album(task.album, False)
        else:
            self.handle_track(task.item, False)

    def commands(self):
        """Return the "replaygain" ui subcommand.
        """
        def func(lib, opts, args):
            write = ui.should_write(opts.write)
            force = opts.force

            if opts.album:
                for album in lib.albums(ui.decargs(args)):
                    self.handle_album(album, write, force)

            else:
                for item in lib.items(ui.decargs(args)):
                    self.handle_track(item, write, force)

        cmd = ui.Subcommand('replaygain', help=u'analyze for ReplayGain')
        cmd.parser.add_album_option()
        cmd.parser.add_option(
            "-f", "--force", dest="force", action="store_true", default=False,
            help=u"analyze all files, including those that "
            "already have ReplayGain metadata")
        cmd.parser.add_option(
            "-w", "--write", default=None, action="store_true",
            help=u"write new metadata to files' tags")
        cmd.parser.add_option(
            "-W", "--nowrite", dest="write", action="store_false",
            help=u"don't write metadata (opposite of -w)")
        cmd.func = func
        return [cmd]
