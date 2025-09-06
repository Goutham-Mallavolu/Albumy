# -*- coding: utf-8 -*-
"""
    :author: Grey Li (李辉)
    :url: http://greyli.com
    :copyright: © 2018 Grey Li <withlihui@gmail.com>
    :license: MIT, see LICENSE for more details.
"""


import os

import click
from flask import Flask, render_template
from flask_login import current_user
from flask_wtf.csrf import CSRFError

from albumy.blueprints.admin import admin_bp
from albumy.blueprints.ajax import ajax_bp
from albumy.blueprints.auth import auth_bp
from albumy.blueprints.main import main_bp
from albumy.blueprints.user import user_bp
from albumy.extensions import bootstrap, db, login_manager, mail, dropzone, moment, whooshee, avatars, csrf
from albumy.models import Role, User, Photo, Tag, Follow, Notification, Comment, Collect, Permission
from albumy.settings import config


def create_app(config_name=None):
    if config_name is None:
        config_name = os.getenv('FLASK_CONFIG', 'development')

    app = Flask('albumy')
    
    app.config.from_object(config[config_name])

    register_extensions(app)
    register_blueprints(app)
    register_commands(app)
    register_errorhandlers(app)
    register_shell_context(app)
    register_template_context(app)

    return app


def register_extensions(app):
    bootstrap.init_app(app)
    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    dropzone.init_app(app)
    moment.init_app(app)
    whooshee.init_app(app)
    avatars.init_app(app)
    csrf.init_app(app)


def register_blueprints(app):
    app.register_blueprint(main_bp)
    app.register_blueprint(user_bp, url_prefix='/user')
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(ajax_bp, url_prefix='/ajax')


def register_shell_context(app):
    @app.shell_context_processor
    def make_shell_context():
        return dict(db=db, User=User, Photo=Photo, Tag=Tag,
                    Follow=Follow, Collect=Collect, Comment=Comment,
                    Notification=Notification)


def register_template_context(app):
    @app.context_processor
    def make_template_context():
        if current_user.is_authenticated:
            notification_count = Notification.query.with_parent(current_user).filter_by(is_read=False).count()
        else:
            notification_count = None
        return dict(notification_count=notification_count)


def register_errorhandlers(app):
    @app.errorhandler(400)
    def bad_request(e):
        return render_template('errors/400.html'), 400

    @app.errorhandler(403)
    def forbidden(e):
        return render_template('errors/403.html'), 403

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return render_template('errors/413.html'), 413

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('errors/500.html'), 500

    @app.errorhandler(CSRFError)
    def handle_csrf_error(e):
        return render_template('errors/400.html', description=e.description), 500


def register_commands(app):
    from albumy.semantic.embedding_service import EmbeddingService
    from albumy.semantic.index_manager import IndexManager
    from albumy.models import Photo, Embedding

    @app.cli.command()
    @click.option('--drop', is_flag=True, help='Create after drop.')
    def initdb(drop):
        """Initialize the database."""
        if drop:
            click.confirm('This operation will delete the database, do you want to continue?', abort=True)
            db.drop_all()
            click.echo('Drop tables.')
        db.create_all()
        click.echo('Initialized database.')

    @app.cli.command()
    def init():
        """Initialize Albumy."""
        click.echo('Initializing the database...')
        db.create_all()

        click.echo('Initializing the roles and permissions...')
        Role.init_role()

        click.echo('Done.')

    @app.cli.command()
    @click.option('--user', default=10, help='Quantity of users, default is 10.')
    @click.option('--follow', default=30, help='Quantity of follows, default is 30.')
    @click.option('--photo', default=30, help='Quantity of photos, default is 30.')
    @click.option('--tag', default=20, help='Quantity of tags, default is 20.')
    @click.option('--collect', default=50, help='Quantity of collects, default is 50.')
    @click.option('--comment', default=100, help='Quantity of comments, default is 100.')
    def forge(user, follow, photo, tag, collect, comment):
        """Generate fake data."""

        from albumy.fakes import fake_admin, fake_comment, fake_follow, fake_photo, fake_tag, fake_user, fake_collect

        db.drop_all()
        db.create_all()

        click.echo('Initializing the roles and permissions...')
        Role.init_role()
        click.echo('Generating the administrator...')
        fake_admin()
        click.echo('Generating %d users...' % user)
        fake_user(user)
        click.echo('Generating %d follows...' % follow)
        fake_follow(follow)
        click.echo('Generating %d tags...' % tag)
        fake_tag(tag)
        click.echo('Generating %d photos...' % photo)
        fake_photo(photo)
        click.echo('Generating %d collects...' % photo)
        fake_collect(collect)
        click.echo('Generating %d comments...' % comment)
        fake_comment(comment)
        click.echo('Done.')

    @app.cli.command()
    @click.option('--rebuild', is_flag=True, help='Rebuild vector index from DB embeddings.')
    def index_embeddings(rebuild):
        """Build or update the semantic vector index from stored embeddings."""
        from albumy.extensions import db
        es = EmbeddingService(model_name=app.config.get('SEMANTIC_MODEL_NAME', 'clip-ViT-B-32'))
        dim = 512 if es._st_model is not None else 384
        idx = IndexManager(dim=dim, index_path=app.config['SEMANTIC_INDEX_PATH'], mapping_path=app.config['SEMANTIC_MAPPING_PATH'])
        # Fetch all embeddings
        all_embs = Embedding.query.all()
        import numpy as np
        if rebuild:
            vectors = []
            ids = []
            for e in all_embs:
                v = Embedding.unpack(e.vector).astype('float32')
                vectors.append(v)
                ids.append(e.photo_id)
            if vectors:
                idx.rebuild(np.vstack(vectors).astype('float32'), np.array(ids, dtype='int64'))
            else:
                idx.rebuild(np.zeros((0, dim), dtype='float32'), np.array([], dtype='int64'))
            idx.save()
            click.echo(f'Rebuilt index with {idx.size} items.')
        else:
            # Incremental add for new ones not in index mapping
            existing = set(idx._ids.tolist())
            add_vecs = []
            add_ids = []
            for e in all_embs:
                if e.photo_id not in existing:
                    add_vecs.append(Embedding.unpack(e.vector).astype('float32'))
                    add_ids.append(e.photo_id)
            if add_vecs:
                import numpy as np
                idx.add(np.vstack(add_vecs).astype('float32'), np.array(add_ids, dtype='int64'))
                idx.save()
            click.echo(f'Indexed {len(add_ids)} new items; total {idx.size}.')

    @app.cli.command()
    def backfill_embeddings():
        """Compute and store embeddings for photos lacking them."""
        from albumy.extensions import db
        es = EmbeddingService(model_name=app.config.get('SEMANTIC_MODEL_NAME', 'clip-ViT-B-32'))
        dim = 512 if es._st_model is not None else 384
        count = 0
        for photo in Photo.query.all():
            if getattr(photo, 'embedding', None) is None:
                img_path = os.path.join(app.config['ALBUMY_UPLOAD_PATH'], photo.filename)
                vec = es.encode_image(img_path)
                emb = Embedding(photo=photo, dim=len(vec), vector=Embedding.pack(vec))
                db.session.add(emb)
                count += 1
        db.session.commit()
        click.echo(f'Backfilled embeddings for {count} photos.')
